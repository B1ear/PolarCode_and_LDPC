"""Check what decoder polarcodes library uses"""
from polarcodes import PolarCode, Construct, Decode
import numpy as np

N, K = 8, 4
pc = PolarCode(N, K)
Construct(pc, design_SNR=2.0)

# Set some LLR
llr = np.array([10.0, -5.0, 8.0, -12.0, 6.0, -9.0, 7.0, -11.0])
pc.likelihoods = llr

# Check available decoders
print("Testing different decoders:")
print()

decoder_names = ['scd', 'scl', 'scan']

for decoder_name in decoder_names:
    try:
        pc_test = PolarCode(N, K)
        Construct(pc_test, design_SNR=2.0)
        pc_test.likelihoods = llr.copy()
        
        Decode(pc_test, decoder_name=decoder_name)
        result = pc_test.message_received
        
        print(f"{decoder_name:10s}: {result}")
    except Exception as e:
        print(f"{decoder_name:10s}: ERROR - {e}")

print()
print("Note: 'scd' is Successive Cancellation Decoder")
print("      'scl' is Successive Cancellation List Decoder") 
print("      'scan' is SCAN decoder")
