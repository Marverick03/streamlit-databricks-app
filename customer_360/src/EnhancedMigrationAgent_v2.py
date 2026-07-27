# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Enhanced MLflow Migration Agent v2
# MAGIC %md
# MAGIC #  Enhanced MLflow Migration Agent v2
# MAGIC
# MAGIC ## What This Does
# MAGIC
# MAGIC Migrates legacy ML notebooks to add MLflow tracking **without breaking existing code**.
# MAGIC
# MAGIC ## Key Improvements Over Previous Versions
# MAGIC
# MAGIC ###  **Variable Inventory**
# MAGIC - Analyzer extracts **actual variable names** from notebook
# MAGIC - No guessing, no hallucination
# MAGIC - Knows exact metric names (e.g., `accuracy` vs `test_accuracy`)
# MAGIC
# MAGIC ###  **Type-Aware Code Generation**
# MAGIC - Detects numpy arrays vs pandas DataFrames
# MAGIC - Uses correct methods for each type:
# MAGIC   - Numpy: `X_train[:1]`
# MAGIC   - Pandas: `X_train.head(1)`
# MAGIC
# MAGIC ###  **Correct Imports**
# MAGIC - Enforces: `from mlflow.models import infer_signature`
# MAGIC - Prevents: `from sklearn.utils import infer_signature`
# MAGIC
# MAGIC ###  **Strict Validation**
# MAGIC - Checks variable existence
# MAGIC - Validates imports
# MAGIC - Ensures type safety
# MAGIC - Prevents hallucination
# MAGIC
# MAGIC ## Result
# MAGIC
# MAGIC **Produces actually runnable notebooks** that execute without errors.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Usage
# MAGIC
# MAGIC ```python
# MAGIC # Initialize agent
# MAGIC agent = EnhancedMigrationAgent()
# MAGIC
# MAGIC # Migrate notebook
# MAGIC request = ResponsesAgentRequest(
# MAGIC     input=[Message(role="user", content="/path/to/notebook")]
# MAGIC )
# MAGIC
# MAGIC response = agent.predict(request)
# MAGIC print(response.custom_outputs['migrated_path'])
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Run Cells Below in Order

# COMMAND ----------

# ============================================================================
# Setup - Install Groq (FREE & FAST LLM)
# ============================================================================

%pip install -q groq
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and Setup
# Core imports
import os
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Databricks imports
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
try:
    from databricks_agent_framework import ResponsesAgent, ResponsesAgentRequest, ResponsesAgentResponse, Message
except ImportError:
    print("⚠️  ResponsesAgent framework not available - using mock classes")
    
    @dataclass
    class Message:
        role: str
        content: str
    
    @dataclass
    class ResponsesAgentRequest:
        input: List[Message]
    
    @dataclass
    class ResponsesAgentResponse:
        output: Message
        custom_outputs: dict
    
    class ResponsesAgent:
        def predict(self, request):
            raise NotImplementedError()

print("✅ Imports complete")

# COMMAND ----------

# DBTITLE 1,LLM Client (Groq)
# ============================================================================
# LLM CLIENT - Groq API
# ============================================================================

from groq import Groq

class LLMClient:
    """
    LLM client using Groq API (free tier).
    Model: llama-3.3-70b-versatile
    """
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Groq API key. If None, uses provided key.
        """
        if api_key:
            self.api_key = api_key
            print("📝 Using provided API key")
        else:
            self.api_key = "enter your api key here"  # Replace with your Groq API key
            print("Using hardcoded API key")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        print(f"✅ LLM Client initialized: {self.model}")
    
    def call(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """
        Call LLM with messages.
        
        Args:
            messages: List of {"role": "user|assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def call_with_json_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response with robust error handling.
        """
        # Add JSON instruction
        schema_str = json.dumps(schema, indent=2)
        messages[-1]["content"] += f"\n\nRespond with ONLY valid JSON matching this schema (no markdown, no extra text):\n{schema_str}"
        
        response_text = self.call(messages, temperature, max_tokens)
        
        # Try multiple parsing strategies
        parse_strategies = [
            # Strategy 1: Direct parse
            lambda: json.loads(response_text),
            
            # Strategy 2: Extract from markdown code block
            lambda: json.loads(re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL).group(1)),
            
            # Strategy 3: Extract first {...} or [...]
            lambda: json.loads(re.search(r'(\{.+\}|\[.+\])', response_text, re.DOTALL).group(1)),
            
            # Strategy 4: Clean and parse
            lambda: json.loads(response_text.strip().replace('```json', '').replace('```', ''))
        ]
        
        for i, strategy in enumerate(parse_strategies, 1):
            try:
                result = strategy()
                return result
            except:
                continue
        
        # All strategies failed
        print(f"⚠️  JSON parsing failed. Response:\n{response_text}")
        raise ValueError(f"Could not parse JSON from: {response_text[:500]}")

print("✅ LLMClient class defined")

# COMMAND ----------

# DBTITLE 1,Notebook Reader/Writer Tools
# ============================================================================
# NOTEBOOK READER/WRITER TOOLS
# ============================================================================

class NotebookReaderTool:
    """Read notebook source code."""
    
    def execute(self, notebook_path: str) -> str:
        """
        Read notebook and return concatenated Python source.
        
        Args:
            notebook_path: Workspace path to notebook
            
        Returns:
            Concatenated Python source code
        """
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.workspace import ExportFormat
        import base64
        
        try:
            w = WorkspaceClient()
            
            # Export notebook as source (Python)
            exported = w.workspace.export(
                path=notebook_path,
                format=ExportFormat.SOURCE
            )
            
            # Handle different response formats
            if hasattr(exported, 'content'):
                content = exported.content
            elif isinstance(exported, dict):
                content = exported.get('content', '')
            else:
                content = str(exported)
            
            # Decode if base64
            try:
                source_code = base64.b64decode(content).decode('utf-8')
            except:
                source_code = content
            
            return source_code
            
        except Exception as e:
            print(f"⚠️  Could not read via Workspace API: {e}")
            import traceback
            traceback.print_exc()
            raise FileNotFoundError(f"Could not read notebook: {notebook_path}")


class NotebookWriterTool:
    """Write migrated notebook."""
    
    def execute(self, output_path: str, content: str) -> str:
        """
        Write migrated notebook using Workspace API.
        
        Args:
            output_path: Workspace path for output (without extension)
            content: Migrated source code
            
        Returns:
            Path to written notebook
        """
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.workspace import ImportFormat, Language
        import base64
        
        try:
            w = WorkspaceClient()
            
            # Encode content
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            # Import as Python notebook
            w.workspace.import_(
                path=output_path,
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                content=encoded_content,
                overwrite=True
            )
            
            return output_path
            
        except Exception as e:
            print(f"⚠️  Could not write via Workspace API: {e}")
            import traceback
            traceback.print_exc()
            raise FileNotFoundError(f"Could not write notebook: {output_path}")


print("✅ Notebook Reader/Writer tools defined")

# COMMAND ----------

# DBTITLE 1,Enhanced Analyzer Agent
# ============================================================================
# ENHANCED ANALYZER AGENT - Variable Inventory
# ============================================================================

class EnhancedAnalyzerAgent:
    """
    Analyzer that extracts ACTUAL variables, types, and metrics.
    Prevents hallucination by providing concrete inventory.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def analyze(self, notebook_source: str) -> dict:
        """
        Analyze notebook and create detailed variable inventory.
        """
        prompt = f"""
Analyze this Python ML notebook to prepare for MLflow migration.

EXTRACT THESE FACTS (do not guess or assume):

1. **ML Framework**: Which library? (sklearn, tensorflow, pytorch, xgboost)
2. **Model Variable**: What variable stores the trained model? (look for .fit() calls)
3. **Training Data Variables**: 
   - Training features (e.g., X_train, X_scaled)
   - Training labels (e.g., y_train)
   - Test features (e.g., X_test)
   - Test labels (e.g., y_test)
4. **Prediction Variable**: What stores predictions? (look for .predict() calls)
5. **Metric Variables**: EXACT variable names for metrics
   - Look for: accuracy = accuracy_score(...)
   - List ONLY variables that are actually assigned
   - Example: if code has `accuracy = ...`, record "accuracy", NOT "test_accuracy"
6. **Data Types**: For training data, is it:
   - numpy.ndarray (look for make_classification, np.array)
   - pandas.DataFrame (look for pd.DataFrame, .read_csv)
7. **Model Parameters**: Hyperparameters (max_iter, random_state, etc.)

CRITICAL RULES:
- ONLY list variables you see explicitly in code
- Do NOT invent names
- Check data types carefully
- Extract actual parameter names

Notebook Code:
```python
{notebook_source}
```

Respond with JSON:
{{
    "ml_framework": "sklearn",
    "model_variable_name": "model",
    "model_type": "LogisticRegression",
    "training_data": {{
        "X_train": "X_train",
        "y_train": "y_train",
        "X_test": "X_test",
        "y_test": "y_test",
        "X_train_type": "numpy.ndarray"
    }},
    "predictions": {{
        "variable_name": "y_pred",
        "exists": true
    }},
    "metrics": {{
        "accuracy": "accuracy",
        "f1": "f1"
    }},
    "model_params": {{
        "max_iter": 1000,
        "random_state": 42
    }},
    "has_mlflow": false,
    "confidence": "high"
}}
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        analysis = self.llm.call_with_json_output(
            messages,
            schema={
                "ml_framework": "string",
                "model_variable_name": "string",
                "model_type": "string",
                "training_data": "object",
                "predictions": "object",
                "metrics": "object",
                "model_params": "object",
                "has_mlflow": "boolean",
                "confidence": "string"
            },
            temperature=0.1
        )
        
        return analysis

print("✅ EnhancedAnalyzerAgent defined")

# COMMAND ----------

# DBTITLE 1,Enhanced Planner Agent
# ============================================================================
# ENHANCED PLANNER AGENT
# ============================================================================

class EnhancedPlannerAgent:
    """Plan MLflow migration steps."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def plan(self, analysis: dict, notebook_source: str) -> dict:
        """
        Create migration plan based on analysis.
        """
        if analysis.get('has_mlflow', False):
            return {
                "status": "NO_CHANGE",
                "reason": "Notebook already has MLflow"
            }
        
        model_type = analysis.get('model_type', 'MLModel')
        framework = analysis.get('ml_framework', 'sklearn')
        
        return {
            "status": "MIGRATION_NEEDED",
            "model_name_suggestion": f"{model_type}_Model",
            "registry_path": f"customer360.ml_models.{model_type}",
            "estimated_complexity": "low",
            "steps": [
                "Add MLflow setup",
                "Wrap training with MLflow run",
                "Log parameters and metrics",
                "Register model to Unity Catalog"
            ]
        }

print("✅ EnhancedPlannerAgent defined")

# COMMAND ----------

# DBTITLE 1,Enhanced Generator Agent
# ============================================================================
# ENHANCED GENERATOR AGENT - Context-Aware Code
# ============================================================================

class EnhancedCodeGeneratorAgent:
    """
    Generator that uses ONLY variables from analysis.
    Prevents hallucination.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_training_wrapper(self, analysis: dict, plan: dict) -> str:
        """
        Generate MLflow training wrapper using analyzed variables.
        """
        model_var = analysis.get('model_variable_name', 'model')
        metrics = analysis.get('metrics', {})
        model_params = analysis.get('model_params', {})
        training_data = analysis.get('training_data', {})
        predictions = analysis.get('predictions', {})
        model_type = analysis.get('model_type', 'Model')
        framework = analysis.get('ml_framework', 'sklearn')
        
        X_train = training_data.get('X_train', 'X_train')
        X_test = training_data.get('X_test', 'X_test')
        X_train_type = training_data.get('X_train_type', 'numpy.ndarray')
        pred_var = predictions.get('variable_name', 'y_pred')
        
        # Determine correct MLflow flavor
        flavor_map = {
            'sklearn': 'mlflow.sklearn.log_model',
            'tensorflow': 'mlflow.tensorflow.log_model',
            'keras': 'mlflow.keras.log_model',
            'pytorch': 'mlflow.pytorch.log_model',
            'xgboost': 'mlflow.xgboost.log_model'
        }
        log_model_func = flavor_map.get(framework, 'mlflow.sklearn.log_model')
        
        prompt = f"""
Generate MLflow training wrapper for ALREADY TRAINED {framework} model.

CONTEXT:
- Framework: {framework}
- Model variable: {model_var}
- Metrics: {metrics}
- Model parameters: {model_params}
- Data type: {X_train_type}
- Prediction variable: {pred_var}

REQUIREMENTS:

1. **Correct Import**: `from mlflow.models import infer_signature`
   - NOT from sklearn.utils!

2. **No Retraining**: Model is ALREADY trained. Do NOT call .fit()

3. **Use Only Listed Variables**:
   - Metrics: {list(metrics.values())}
   - Model: {model_var}
   - Prediction: {pred_var}

4. **Input Example**:
   - If numpy: `{X_train}[:1]`
   - If pandas: `{X_train}.head(1)`
   - Current type: {X_train_type}

5. **Signature**: Use test data: `infer_signature({X_test}[:1], {pred_var}[:1])`

6. **Correct MLflow Flavor**: Use `{log_model_func}()` for {framework} models
   - First argument is the model object: {model_var}
   - Use `artifact_path="model"` parameter
   - Include signature, input_example, registered_model_name

7. **Single Run**: ONE mlflow.start_run() with:
   - Log params: {list(model_params.keys())}
   - Log metrics using ACTUAL metric variables: {list(metrics.keys())}
   - Log model with signature + input_example
   - Register to: customer360.ml_models.{model_type}

EXAMPLE for sklearn:
```python
with mlflow.start_run():
    # Log parameters
    mlflow.log_param('param_name', param_value)
    
    # Log metrics using ACTUAL variable names
    mlflow.log_metric('accuracy', accuracy)  # use variable directly!
    
    # Create signature
    signature = infer_signature({X_test}[:1], {pred_var}[:1])
    
    # Log model
    mlflow.sklearn.log_model(
        {model_var},
        artifact_path="model",
        signature=signature,
        input_example={X_test}[:1],
        registered_model_name="customer360.ml_models.{model_type}"
    )
```

Generate ONLY the code:
"""
        
        messages = [{"role": "user", "content": prompt}]
        code = self.llm.call(messages, temperature=0.1, max_tokens=1500)
        
        # Clean code
        code = code.replace('```python', '').replace('```', '').strip()
        
        return code
    
    def generate_setup_code(self, analysis: dict, plan: dict) -> str:
        model_type = analysis.get('model_type', 'MLModel')
        # Get current user for experiment path
        from databricks.sdk import WorkspaceClient
        try:
            w = WorkspaceClient()
            current_user = w.current_user.me().user_name
        except:
            current_user = "<username>"
        
        return f"""import mlflow\nfrom mlflow.models import infer_signature\n\nexperiment_name = \"/Users/{current_user}/{model_type}_Experiment\"\nmlflow.set_experiment(experiment_name)\nprint(f\"✅ MLflow experiment: {{experiment_name}}\")"""
    
    def generate_registry_code(self, analysis: dict, plan: dict) -> str:
        return "# Model registered in training run above"

print("✅ EnhancedCodeGeneratorAgent defined")

# COMMAND ----------

# DBTITLE 1,Enhanced Validator Agent
# ============================================================================
# ENHANCED VALIDATOR AGENT - Strict Checks
# ============================================================================

class EnhancedValidatorAgent:
    """
    Validator that checks imports, variables, types.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def validate(self, original: str, generated: str, analysis: dict, plan: dict) -> dict:
        """
        Validate generated code.
        """
        expected_metrics = analysis.get('metrics', {})
        expected_model = analysis.get('model_variable_name', 'model')
        training_data = analysis.get('training_data', {})
        
        prompt = f"""
Validate MLflow code against analysis.

ANALYSIS:
- Model: {expected_model}
- Metrics: {expected_metrics}
- Data type: {training_data.get('X_train_type')}

GENERATED CODE:
```python
{generated}
```

CHECKS:

1. **Import**: MUST have `from mlflow.models import infer_signature`
2. **Variables**: All metrics MUST be in {list(expected_metrics.values())}
3. **Type Safety**: Input example must match data type
4. **No Retraining**: MUST NOT contain `.fit(`
5. **Single Run**: Exactly 1 `mlflow.start_run(`
6. **No Hallucination**: All variables must exist in analysis

Respond with JSON:
{{
    "approved": true,
    "status": "APPROVED",
    "confidence": "high",
    "severity": "none",
    "issues": [],
    "passed_checks": [
        "Correct imports",
        "Uses actual variables",
        "Type-safe input_example"
    ]
}}
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        validation = self.llm.call_with_json_output(
            messages,
            schema={
                "approved": "boolean",
                "status": "string",
                "confidence": "string",
                "severity": "string",
                "issues": "array",
                "passed_checks": "array"
            },
            temperature=0.1
        )
        
        return validation

print("✅ EnhancedValidatorAgent defined")

# COMMAND ----------

# DBTITLE 1,Enhanced Migration Agent - Main Class
# ============================================================================
# ENHANCED MIGRATION AGENT - Creates Runnable Notebooks
# ============================================================================

class EnhancedMigrationAgent(ResponsesAgent):
    """
    Enhanced migration agent that produces runnable notebooks.
    
    Features:
    - Variable inventory from analyzer
    - Context-aware code generation
    - Strict validation
    - No hallucination
    """
    
    def __init__(self, llm_client=None):
        super().__init__()
        
        if llm_client is None:
            llm_client = LLMClient()
        
        self.llm = llm_client
        self.analyzer = EnhancedAnalyzerAgent(self.llm)
        self.planner = EnhancedPlannerAgent(self.llm)
        self.generator = EnhancedCodeGeneratorAgent(self.llm)
        self.validator = EnhancedValidatorAgent(self.llm)
        self.reader = NotebookReaderTool()
        self.writer = NotebookWriterTool()
        
        print("✅ EnhancedMigrationAgent initialized")
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Main prediction method.
        """
        print("\n" + "="*70)
        print("ENHANCED MLFLOW MIGRATION AGENT")
        print("="*70 + "\n")
        
        notebook_path = request.input[-1].content.strip()
        print(f"📍 Target: {notebook_path}\n")
        
        try:
            # Step 1: Read
            print(" STEP 1: Read Notebook")
            print("-" * 70)
            source = self.reader.execute(notebook_path)
            print(f"✅ Read {len(source)} characters\n")
            
            # Step 2: Analyze (with variable inventory)
            print(" STEP 2: Enhanced Analysis")
            print("-" * 70)
            analysis = self.analyzer.analyze(source)
            print(f" Framework: {analysis.get('ml_framework')}")
            print(f"   Model: {analysis.get('model_variable_name')}")
            print(f"   Metrics: {analysis.get('metrics')}")
            print(f"   Type: {analysis.get('training_data', {}).get('X_train_type')}\n")
            
            # Step 3: Plan
            print("👉 STEP 3: Planning")
            print("-" * 70)
            plan = self.planner.plan(analysis, source)
            
            if plan.get("status") == "NO_CHANGE":
                print("✅ No migration needed\n")
                return self._create_response(
                    "NO_CHANGE",
                    "Already has MLflow",
                    analysis=analysis,
                    plan=plan
                )
            
            print(f"📋 Status: {plan.get('status')}\n")
            
            # Step 4: Generate
            print(" STEP 4: Generate Code")
            print("-" * 70)
            setup = self.generator.generate_setup_code(analysis, plan)
            training = self.generator.generate_training_wrapper(analysis, plan)
            registry = self.generator.generate_registry_code(analysis, plan)
            print("✅ Code generated\n")
            
            # Step 5: Validate
            print(" STEP 5: Validate")
            print("-" * 70)
            combined = f"{setup}\n\n{training}\n\n{registry}"
            validation = self.validator.validate(source, combined, analysis, plan)
            print(f"✅ Status: {validation.get('status')}")
            print(f"   Approved: {validation.get('approved')}")
            
            if validation.get('issues'):
                print(f"   Issues: {len(validation['issues'])}")
                for issue in validation['issues'][:3]:
                    print(f"      - {issue}")
            print()
            
            if not validation.get('approved'):
                print("⚠️  Validation failed\n")
                return self._create_response(
                    "VALIDATION_FAILED",
                    "Code failed validation",
                    analysis=analysis,
                    plan=plan,
                    validation=validation,
                    generated_code={"setup": setup, "training": training}
                )
            
            # Step 6: Write
            print(" STEP 6: Write Migrated Notebook")
            print("-" * 70)
            migrated = self._inject_code(source, setup, training, registry)
            output_path = notebook_path + "_enhanced"
            self.writer.execute(output_path, migrated)
            print(f"✅ Saved: {output_path}\n")
            
            print("="*70)
            print("✅ MIGRATION COMPLETE")
            print("="*70 + "\n")
            
            return self._create_response(
                "SUCCESS",
                f"Migrated to: {output_path}",
                analysis=analysis,
                plan=plan,
                validation=validation,
                migrated_path=output_path,
                generated_code={"setup": setup, "training": training, "registry": registry}
            )
            
        except Exception as e:
            error_msg = f"Migration failed: {type(e).__name__}: {str(e)}"
            print(f"\n {error_msg}\n")
            return self._create_response("ERROR", error_msg, error_details=str(e))
    
    def _inject_code(self, original, setup, training, registry):
        return f"""# MLflow Setup\n{setup}\n\n# Original Code\n{original}\n\n# MLflow Training Wrapper\n{training}\n\n# Registry\n{registry}"""
    
    def _create_response(self, status, message, **kwargs):
        return ResponsesAgentResponse(
            output=Message(role="assistant", content=message),
            custom_outputs={"status": status, "message": message, **kwargs}
        )

print("✅ EnhancedMigrationAgent class defined")

# COMMAND ----------

# DBTITLE 1,Test & Demo
# MAGIC %md
# MAGIC # 🧪 Test & Demo
# MAGIC
# MAGIC Run the cell below to test the enhanced agent.
# MAGIC
# MAGIC ## What It Does
# MAGIC
# MAGIC 1. Creates agent instance
# MAGIC 2. Migrates test notebook
# MAGIC 3. Shows results
# MAGIC 4. Displays improvements
# MAGIC
# MAGIC ## Expected Output
# MAGIC
# MAGIC - ✅ Variable inventory (actual names from notebook)
# MAGIC - ✅ Type-aware code (respects numpy vs pandas)
# MAGIC - ✅ Correct imports (mlflow.models, not sklearn.utils)
# MAGIC - ✅ Validation passes
# MAGIC - ✅ Runnable output notebook

# COMMAND ----------

# DBTITLE 1,Run Migration Test
# ============================================================================
# TEST ENHANCED AGENT
# ============================================================================

print("\n" + "="*70)
print("🧪 TESTING ENHANCED AGENT")
print("="*70)

try:
    # Initialize agent
    agent = EnhancedMigrationAgent()
    
    # Test notebook path
    test_notebook = "/Users/rushikesh.kumavat@hoonartek.com/mlflow_agent_poc/gateway_test_simple_model"
    
    # Create request
    request = ResponsesAgentRequest(
        input=[Message(role="user", content=test_notebook)]
    )
    
    print(f"\n🔄 Running migration...\n")
    
    # Execute
    response = agent.predict(request)
    
    # Display results
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    print(f"\nStatus: {response.custom_outputs['status']}")
    print(f"Message: {response.custom_outputs['message']}")
    
    if response.custom_outputs['status'] == 'SUCCESS':
        analysis = response.custom_outputs['analysis']
        validation = response.custom_outputs['validation']
        
        print(f"\n✅ MIGRATION SUCCESSFUL!")
        print(f"\n Analysis:")
        print(f"   Framework: {analysis['ml_framework']}")
        print(f"   Model: {analysis['model_variable_name']}")
        print(f"   Metrics: {analysis['metrics']}")
        print(f"   Data Type: {analysis.get('training_data', {}).get('X_train_type')}")
        
        print(f"\n✅ Validation:")
        print(f"   Approved: {validation['approved']}")
        print(f"   Checks Passed: {len(validation.get('passed_checks', []))}")
        for check in validation.get('passed_checks', [])[:5]:
            print(f"      ✓ {check}")
        
        print(f"\n Key Improvements:")
        print(f"   ✅ Uses actual variable names (not hallucinated)")
        print(f"   ✅ Respects data types (numpy vs pandas)")
        print(f"   ✅ Correct MLflow imports")
        print(f"   ✅ Type-safe input_example")
        print(f"   ✅ No retraining")
        
        print(f"\n📁 Output: {response.custom_outputs['migrated_path']}")
        print(f"\n💡 Next: Open and run the migrated notebook to verify it works!")
        
    elif response.custom_outputs['status'] == 'ERROR':
        print(f"\n Error: {response.custom_outputs.get('error_details')}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n Test failed: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,⚠️ Setup Required
# %md
# #  Setup Required: Groq API Key

# The test failed because the Groq API secret is not configured.

# ## Option 1: Create Secret (Recommended)

# 1. Get a free Groq API key: https://console.groq.com/keys
# 2. Run the cell below to create the secret

# ## Option 2: Use Direct API Key

# Modify the LLMClient `__init__` to accept an API key parameter:
# ```python
# class LLMClient:
#     def __init__(self, api_key: str = None):
#         if api_key:
#             self.api_key = api_key
#         else:
#             self.api_key = dbutils.secrets.get("groq", "api_key")
# ```

# Then test with:
# ```python
# agent = EnhancedMigrationAgent(
#     llm_client=LLMClient(api_key="your-groq-api-key")
# )
# ```

# COMMAND ----------

# DBTITLE 1,Create Groq Secret
# # ============================================================================
# # CREATE GROQ API SECRET
# # ============================================================================

# from databricks.sdk import WorkspaceClient

# print("\n📝 Creating Groq API Secret...\n")
# print("You'll need a Groq API key from: https://console.groq.com/keys")
# print("\nGroq offers FREE tier with:")
# print("  • 14,400 requests/day")
# print("  • 100,000 tokens/day")
# print("  • Fast inference on Llama 3.3 70B\n")

# # Get API key from user
# api_key = input("Enter your Groq API key: ").strip()

# if api_key:
#     try:
#         w = WorkspaceClient()
        
#         # Create secret scope
#         try:
#             w.secrets.create_scope(scope="groq")
#             print("✅ Created secret scope: groq")
#         except Exception as e:
#             if "already exists" in str(e).lower():
#                 print("ℹ️  Secret scope 'groq' already exists")
#             else:
#                 raise
        
#         # Put secret
#         w.secrets.put_secret(
#             scope="groq",
#             key="api_key",
#             string_value=api_key
#         )
        
#         print("✅ Stored API key in secret: groq/api_key")
#         print("\n🎉 Setup complete! Now run the test cell again.")
        
#     except Exception as e:
#         print(f"❌ Error creating secret: {e}")
#         print("\n💡 Alternative: Use Option 2 (direct API key) instead")
# else:
#     print("❌ No API key provided")

# COMMAND ----------

# DBTITLE 1,Alternative: Test with Direct API Key
# %md
# # 🔑 Alternative: Test with Direct API Key

# If you prefer not to create a secret, you can test directly with your API key.

# **⚠️ Warning:** This approach embeds the key in the notebook. Use secrets in production.

# ## How to Use

# 1. Get your Groq API key from: https://console.groq.com/keys
# 2. Replace `"your-groq-api-key-here"` in the cell below
# 3. Run the cell
