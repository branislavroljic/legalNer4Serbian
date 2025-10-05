"""
Debug script to inspect training history structure.
Run this in your notebook after training to see what's in the training_history.
"""

def inspect_training_history(fold_results):
    """
    Inspect the structure of training history to debug visualization issues.
    
    Args:
        fold_results: List of fold results with training_history
    """
    print("="*80)
    print("INSPECTING TRAINING HISTORY STRUCTURE")
    print("="*80)
    
    for fold_idx, fold_result in enumerate(fold_results, 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*80}")
        
        if 'training_history' not in fold_result:
            print("❌ No 'training_history' field found!")
            continue
        
        history = fold_result['training_history']
        print(f"✅ Training history found with {len(history)} entries")
        
        # Count different types of entries
        loss_only = 0
        eval_only = 0
        both = 0
        other = 0
        
        for entry in history:
            has_loss = 'loss' in entry and 'eval_loss' not in entry
            has_eval = 'eval_loss' in entry
            
            if has_loss and not has_eval:
                loss_only += 1
            elif has_eval and not has_loss:
                eval_only += 1
            elif has_loss and has_eval:
                both += 1
            else:
                other += 1
        
        print(f"\nEntry types:")
        print(f"  - Training loss only: {loss_only}")
        print(f"  - Evaluation metrics only: {eval_only}")
        print(f"  - Both: {both}")
        print(f"  - Other: {other}")
        
        # Show first few entries of each type
        print(f"\nSample entries:")
        
        # Find first training loss entry
        for entry in history:
            if 'loss' in entry and 'eval_loss' not in entry:
                print(f"\n  Training loss entry:")
                print(f"    Keys: {list(entry.keys())}")
                print(f"    Values: {entry}")
                break
        
        # Find first evaluation entry
        for entry in history:
            if 'eval_loss' in entry:
                print(f"\n  Evaluation entry:")
                print(f"    Keys: {list(entry.keys())}")
                print(f"    Sample values:")
                for key in ['step', 'epoch', 'eval_loss', 'eval_precision', 'eval_recall', 'eval_f1']:
                    if key in entry:
                        print(f"      {key}: {entry[key]}")
                break
        
        # Check if eval metrics have the required fields
        eval_entries = [e for e in history if 'eval_loss' in entry]
        if eval_entries:
            first_eval = eval_entries[0]
            required_fields = ['eval_precision', 'eval_recall', 'eval_f1', 'eval_accuracy']
            missing_fields = [f for f in required_fields if f not in first_eval]
            
            if missing_fields:
                print(f"\n  ⚠️  WARNING: Evaluation entries missing fields: {missing_fields}")
            else:
                print(f"\n  ✅ All required evaluation fields present")
    
    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    print("This is a debug utility. Import and call inspect_training_history(fold_results) in your notebook.")

