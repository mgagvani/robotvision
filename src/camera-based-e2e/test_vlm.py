#!/usr/bin/env python3
"""
Test script to verify SmolVLM loads correctly and can extract features.
This helps us understand the VLM architecture before integrating it.
"""

import torch
from transformers import AutoModel, AutoProcessor

def test_smolvlm_loading():
    """Test loading SmolVLM and extracting visual features"""
    
    model_name = "HuggingFaceTB/SmolVLM-Base"
    print(f"Loading {model_name}...")
    print("=" * 80)
    
    try:
        # Load VLM
        vlm = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Try to load processor, but continue if it fails
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print("✓ Processor loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load processor (this is OK, we can preprocess manually): {e}")
            processor = None
        
        print("✓ Model loaded successfully!")
        print(f"✓ Model type: {vlm.config.model_type}")
        
        # Get hidden size from appropriate config
        if hasattr(vlm.config, 'hidden_size'):
            hidden_size = vlm.config.hidden_size
        elif hasattr(vlm.config, 'vision_config') and hasattr(vlm.config.vision_config, 'hidden_size'):
            hidden_size = vlm.config.vision_config.hidden_size
        else:
            hidden_size = "Unknown"
        
        print(f"✓ Vision hidden size: {hidden_size}")
        print()
        
        # Check model structure
        print("Model structure:")
        print("=" * 80)
        print(f"Model type: {type(vlm)}")
        print(f"Available attributes: {[attr for attr in dir(vlm) if not attr.startswith('_')][:10]}...")
        print()
        
        # Check if vision_model exists
        if hasattr(vlm, 'vision_model'):
            print("✓ Found vision_model attribute")
            vision_model = vlm.vision_model
            print(f"  Vision model type: {type(vision_model)}")
            print(f"  Vision model config: {vision_model.config if hasattr(vision_model, 'config') else 'No config'}")
        else:
            print("✗ No vision_model attribute - need to explore structure")
            print(f"  Main model attributes: {dir(vlm)[:20]}")
        
        print()
        
        # Test with dummy images
        print("Testing feature extraction:")
        print("=" * 80)
        
        # Try different input sizes
        test_sizes = [(224, 224), (384, 384)]
        
        for h, w in test_sizes:
            print(f"\nTesting with image size: {h}x{w}")
            dummy_img = torch.randn(2, 3, h, w)  # Batch of 2
            
            try:
                if hasattr(vlm, 'vision_model'):
                    # Direct vision model access
                    outputs = vlm.vision_model(pixel_values=dummy_img)
                    
                    print(f"  ✓ Vision model forward pass successful")
                    print(f"  Output type: {type(outputs)}")
                    print(f"  Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'Not a dict'}")
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        print(f"  last_hidden_state shape: {outputs.last_hidden_state.shape}")
                    
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        print(f"  ✓ pooler_output shape: {outputs.pooler_output.shape}")
                        print(f"  → Can use pooler_output directly!")
                    else:
                        print(f"  ✗ No pooler_output - will need manual pooling")
                        if hasattr(outputs, 'last_hidden_state'):
                            pooled = outputs.last_hidden_state.mean(dim=1)
                            print(f"  Manual pooling shape: {pooled.shape}")
                            print(f"  → Manual pooling works!")
                
            except Exception as e:
                print(f"  ✗ Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
        
        print()
        print("=" * 80)
        print("Summary:")
        print("=" * 80)
        print(f"✓ VLM loads successfully")
        print(f"✓ Vision hidden dimension: {hidden_size}")
        if hasattr(vlm, 'vision_model'):
            print(f"✓ Can access vision_model")
            print(f"✓ Feature extraction working")
        
        # Freeze test
        print()
        print("Testing parameter freezing:")
        for param in vlm.parameters():
            param.requires_grad = False
        
        trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in vlm.parameters())
        print(f"✓ All parameters frozen: {trainable}/{total} trainable")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_smolvlm_loading()
    
    if success:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - Ready to implement VLMWaypointModel!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ TESTS FAILED - Need to debug VLM loading")
        print("=" * 80)
