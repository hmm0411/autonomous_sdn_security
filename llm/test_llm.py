from llm.llm_service import call_llm

print("START TEST")

result = call_llm("Explain SDN in simple terms")

print("RESULT:")
print(result)

print("END TEST")