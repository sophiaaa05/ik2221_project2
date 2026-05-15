PROJECT_DIR := /home/jovyan/ik2221_project2
PYTHON      := $(PROJECT_DIR)/venv/bin/python
FRONTEND    := lmcache-vllm-extended/frontend
RESULTS     := $(PROJECT_DIR)/results

.PHONY: server engine clean \
        cache-0 cache-01 cache-05 cache-1 cache-2 cache-4 cache-8

$(RESULTS):
	mkdir -p $(RESULTS)

# ---------------- CLEAN ----------------
clean:
	rm -rf $(PROJECT_DIR)/data/*
	@echo "Cache data cleared."

# ---------------- SERVER ----------------
server:
	$(PYTHON) -m lmcache_server.server 127.0.0.1 65432 $(PROJECT_DIR)/data/

# ---------------- ENGINE ----------------
engine:
	LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml \
	CUDA_VISIBLE_DEVICES=0 \
	$(PYTHON) lmcache-vllm-extended/lmcache_vllm/script.py serve \
		Qwen/Qwen2.5-1.5B-Instruct \
		--gpu-memory-utilization 0.8 \
		--dtype half \
		--port 8000 \
		--guided-decoding-backend lm-format-enforcer

# ---------------- TEST RUNNER ----------------
# define run_test
# 	$(PYTHON) lmcache-vllm-extended/request_generator.py \
# 		--mode scheduled  \
# 		--cache-label $(1) \
# 		--context-dir lmcache-vllm-extended/frontend/data \
# 		--output-dir $(RESULTS)
# endef

define run_test
	RESULTS_LABEL=$(1) $(PYTHON) lmcache-vllm-extended/request_generator_task3.py --n-docs 14
endef


# ---------------- EXPERIMENTS ----------------
cache-0:   $(RESULTS)
	$(call run_test,cache_0gb)

cache-01:  $(RESULTS)
	$(call run_test,cache_01gb)

cache-05:  $(RESULTS)
	$(call run_test,cache_05gb)

cache-1:   $(RESULTS)
	$(call run_test,cache_1gb)

cache-2:   $(RESULTS)
	$(call run_test,cache_2gb)

cache-4:   $(RESULTS)
	$(call run_test,cache_4gb)

cache-8:   $(RESULTS)
	$(call run_test,cache_8gb)