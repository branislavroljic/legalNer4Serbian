#!/usr/bin/env python3
"""
GLiNER Zero-Shot Evaluation Runner for Serbian Legal Documents

This script provides a command-line interface to run GLiNER evaluation
with different models and confidence thresholds.

Usage:
    python run_gliner_evaluation.py --model urchade/gliner_mediumv2.1 --threshold 0.3
    python run_gliner_evaluation.py --model urchade/gliner_large --threshold 0.4
    python run_gliner_evaluation.py --model knowledgator/gliner-multitask-large-v0.5 --threshold 0.2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter, defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gliner import GLiNER
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import classification_report
    print("✅ All dependencies loaded successfully!")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install gliner scikit-learn tqdm numpy")
    sys.exit(1)

# Configuration
LABELSTUDIO_JSON_PATH = "annotations.json"
JUDGMENTS_DIR = "labelstudio_files"

# Serbian Legal Entity Types for GLiNER
LEGAL_ENTITY_TYPES = [
    "court",           # COURT - court institutions
    "judge",           # JUDGE - judge names
    "defendant",       # DEFENDANT - defendant entities
    "prosecutor",      # PROSECUTOR - prosecutor entities
    "registrar",       # REGISTRAR - court registrar
    "case number",     # CASE_NUMBER - case identifiers
    "criminal act",    # CRIMINAL_ACT - criminal acts/charges
    "legal provision", # PROVISION - legal provisions
    "decision date",   # DECISION_DATE - dates of legal decisions
    "sanction",        # SANCTION - sanctions/penalties
    "sanction type",   # SANCTION_TYPE - type of sanction
    "sanction value",  # SANCTION_VALUE - value/duration of sanction
    "procedure costs"  # PROCEDURE_COSTS - legal procedure costs
]

# Mapping from GLiNER labels to ground truth labels
GLINER_TO_GT_MAPPING = {
    "court": "COURT",
    "judge": "JUDGE",
    "defendant": "DEFENDANT",
    "prosecutor": "PROSECUTOR",
    "registrar": "REGISTRAR",
    "case number": "CASE_NUMBER",
    "criminal act": "CRIMINAL_ACT",
    "legal provision": "PROVISION",
    "decision date": "DECISION_DATE",
    "sanction": "SANCTION",
    "sanction type": "SANCTION_TYPE",
    "sanction value": "SANCTION_VALUE",
    "procedure costs": "PROCEDURE_COSTS"
}

# Available GLiNER models
AVAILABLE_MODELS = {
    "medium": "urchade/gliner_mediumv2.1",
    "large": "urchade/gliner_large",
    "multitask": "knowledgator/gliner-multitask-large-v0.5",
    "small": "urchade/gliner_small-v2.1",
    "base": "urchade/gliner_base"
}

class GroundTruthLoader:
    """Load and prepare ground truth data from LabelStudio annotations"""
    
    def __init__(self, judgments_dir: str):
        self.judgments_dir = Path(judgments_dir)
        self.entity_types = set()
        
    def load_text_file(self, file_path: str) -> str:
        """Load text content from judgment file"""
        # Handle different file path formats from LabelStudio
        if "/" in file_path:
            filename = file_path.split("/")[-1]
        else:
            filename = file_path
            
        full_path = self.judgments_dir / filename
        
        if not full_path.exists():
            return ""
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return ""
    
    def extract_ground_truth_entities(self, labelstudio_data: List[Dict]) -> List[Dict]:
        """Extract ground truth entities from LabelStudio annotations"""
        ground_truth_examples = []
        
        for item in tqdm(labelstudio_data, desc="Loading ground truth"):
            file_path = item.get("file_upload", "")
            text_content = self.load_text_file(file_path)
            
            if not text_content:
                continue
            
            annotations = item.get("annotations", [])
            
            for annotation in annotations:
                entities = []
                result = annotation.get("result", [])
                
                for res in result:
                    if res.get("type") == "labels":
                        value = res["value"]
                        start = value["start"]
                        end = value["end"]
                        labels = value["labels"]
                        
                        for label in labels:
                            self.entity_types.add(label)
                            entities.append({
                                'text': text_content[start:end],
                                'label': label,
                                'start': start,
                                'end': end
                            })
                
                if entities:  # Only include documents with entities
                    ground_truth_examples.append({
                        'text': text_content,
                        'entities': entities,
                        'file_path': file_path
                    })
        
        return ground_truth_examples

class GLiNERZeroShotEvaluator:
    """GLiNER Zero-Shot NER Evaluator for Serbian Legal Documents"""
    
    def __init__(self, model_name: str, confidence_threshold: float = 0.3):
        print(f"🌟 Initializing GLiNER: {model_name}")
        
        try:
            self.model = GLiNER.from_pretrained(model_name)
            self.model_name = model_name
            self.confidence_threshold = confidence_threshold
            print(f"✅ GLiNER model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading GLiNER model {model_name}: {e}")
            self.model = None
    
    def predict_entities(self, text: str, entity_types: List[str]) -> List[Dict]:
        """Predict entities using GLiNER zero-shot approach"""
        if self.model is None:
            return []
            
        try:
            entities = self.model.predict_entities(
                text, 
                entity_types, 
                threshold=self.confidence_threshold
            )
            
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    'text': entity['text'],
                    'label': entity['label'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': entity['score'],
                    'method': 'gliner_zero_shot',
                    'model': self.model_name
                })
            
            return sorted(formatted_entities, key=lambda x: x['start'])
            
        except Exception as e:
            print(f"❌ Error in GLiNER prediction: {e}")
            return []
    
    def evaluate_on_dataset(self, ground_truth_examples: List[Dict], 
                           entity_types: List[str], 
                           max_examples: int = None) -> Dict:
        """Evaluate GLiNER on the ground truth dataset"""
        print(f"\n🧪 Starting GLiNER Zero-Shot Evaluation")
        print("=" * 60)
        
        if self.model is None:
            return {'error': 'GLiNER model not loaded'}
        
        # Limit examples if specified
        eval_examples = ground_truth_examples[:max_examples] if max_examples else ground_truth_examples
        
        detailed_results = []
        prediction_counts = Counter()
        confidence_scores = []
        
        print(f"📊 Evaluating on {len(eval_examples)} examples...")
        
        start_time = time.time()
        
        for i, example in enumerate(tqdm(eval_examples, desc="GLiNER Evaluation")):
            text = example['text']
            true_entities = example['entities']
            
            # Get GLiNER predictions
            pred_entities = self.predict_entities(text, entity_types)
            
            # Count predictions by type
            for entity in pred_entities:
                prediction_counts[entity['label']] += 1
                confidence_scores.append(entity['confidence'])
            
            # Store detailed results
            detailed_results.append({
                'example_id': i,
                'text': text[:200] + "..." if len(text) > 200 else text,
                'file_path': example['file_path'],
                'true_entities': true_entities,
                'pred_entities': pred_entities,
                'true_count': len(true_entities),
                'pred_count': len(pred_entities)
            })
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # Calculate statistics
        total_true = sum(len(r['true_entities']) for r in detailed_results)
        total_pred = sum(len(r['pred_entities']) for r in detailed_results)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        results = {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'detailed_results': detailed_results,
            'total_true_entities': total_true,
            'total_pred_entities': total_pred,
            'prediction_counts': dict(prediction_counts),
            'examples_evaluated': len(eval_examples),
            'avg_confidence': avg_confidence,
            'evaluation_time': evaluation_time,
            'entities_per_second': total_pred / evaluation_time if evaluation_time > 0 else 0
        }
        
        print(f"✅ GLiNER evaluation complete!")
        print(f"  📊 True entities: {total_true}")
        print(f"  🤖 Predicted entities: {total_pred}")
        print(f"  ⚡ Average confidence: {avg_confidence:.3f}")
        print(f"  ⏱️ Evaluation time: {evaluation_time:.2f}s")
        print(f"  🚀 Entities/second: {results['entities_per_second']:.2f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run GLiNER Zero-Shot Evaluation on Serbian Legal Documents')
    parser.add_argument('--model', type=str, default='medium',
                       help=f'GLiNER model to use. Options: {list(AVAILABLE_MODELS.keys())} or full model name')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Confidence threshold for GLiNER predictions (default: 0.3)')
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Maximum number of examples to evaluate (default: all)')
    parser.add_argument('--output', type=str, default='gliner_results.json',
                       help='Output file for results (default: gliner_results.json)')
    
    args = parser.parse_args()
    
    # Resolve model name
    if args.model in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[args.model]
        print(f"🎯 Using predefined model: {args.model} -> {model_name}")
    else:
        model_name = args.model
        print(f"🎯 Using custom model: {model_name}")
    
    # Check if required files exist
    if not os.path.exists(LABELSTUDIO_JSON_PATH):
        print(f"❌ Error: {LABELSTUDIO_JSON_PATH} not found!")
        print("Please ensure the annotations file is in the current directory.")
        return 1
    
    if not os.path.exists(JUDGMENTS_DIR):
        print(f"❌ Error: {JUDGMENTS_DIR} directory not found!")
        print("Please ensure the labelstudio_files directory exists.")
        return 1
    
    # Load data
    print("📂 Loading LabelStudio annotations...")
    with open(LABELSTUDIO_JSON_PATH, 'r', encoding='utf-8') as f:
        labelstudio_data = json.load(f)
    
    print(f"✅ Loaded {len(labelstudio_data)} annotated documents")
    
    # Load ground truth
    gt_loader = GroundTruthLoader(JUDGMENTS_DIR)
    ground_truth_examples = gt_loader.extract_ground_truth_entities(labelstudio_data)
    
    print(f"📊 Ground truth: {len(ground_truth_examples)} examples")
    
    # Initialize evaluator
    evaluator = GLiNERZeroShotEvaluator(model_name, args.threshold)
    
    # Run evaluation
    results = evaluator.evaluate_on_dataset(
        ground_truth_examples=ground_truth_examples,
        entity_types=LEGAL_ENTITY_TYPES,
        max_examples=args.max_examples
    )
    
    if 'error' in results:
        print(f"❌ Evaluation failed: {results['error']}")
        return 1
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {args.output}")
    
    # Print summary
    print("\n🎉 GLiNER Zero-Shot Evaluation Complete!")
    print("=" * 60)
    print(f"📊 Model: {results['model_name']}")
    print(f"📄 Documents evaluated: {results['examples_evaluated']}")
    print(f"📈 Coverage ratio: {results['total_pred_entities'] / max(results['total_true_entities'], 1):.3f}")
    print(f"⚡ Avg confidence: {results['avg_confidence']:.3f}")
    print(f"⏱️ Processing speed: {results['entities_per_second']:.2f} entities/sec")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
