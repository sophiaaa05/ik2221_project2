PROJECT_DIR := /home/jovyan/shared/ik2221_project2
PYTHON      := $(PROJECT_DIR)/venv/bin/python
DATA_SRC    := lmcache-vllm-extended/frontend/data
DATA_LIVE   := $(PROJECT_DIR)/data/docs
RESULTS     := $(PROJECT_DIR)/results

.PHONY: server engine clean \
        cache-0 cache-01 cache-05 cache-1 cache-2 cache-4 cache-8 \
        ndocs-2 ndocs-4 ndocs-7 ndocs-10 ndocs-14

$(RESULTS):
	mkdir -p $(RESULTS)

# ── Clean KV cache (run between every experiment) ───────────────────────────
clean:
	rm -rf $(PROJECT_DIR)/data/*
	@echo "Cache cleared."

# ── Server + engine (start once per session, restart between ndocs runs) ────
server:
	$(PYTHON) -m lmcache_server.server 127.0.0.1 65432 $(DATA_LIVE)

engine:
	LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml \
    CUDA_VISIBLE_DEVICES=0 \
	$(PYTHON) lmcache-vllm-extended/lmcache_vllm/script.py serve \
		Qwen/Qwen2.5-1.5B-Instruct \
		--gpu-memory-utilization 0.8 \
		--dtype half \
		--port 8000 \
		--guided-decoding-backend lm-format-enforcer

# ── Stage N docs into the live directory ─────────────────────────────────── 
define stage_docs
	rm -rf $(DATA_LIVE) && mkdir -p $(DATA_LIVE)
	ls $(DATA_SRC)/*.txt | sort | head -$(1) | xargs -I{} cp {} $(DATA_LIVE)/
	@echo "Staged $(1) docs into $(DATA_LIVE)."
endef

# ── Cache-size experiments (all 14 docs, vary cache config in yaml) ──────── 
define run_cache_test
	RESULTS_LABEL=$(1) RESULTS_DIR=$(RESULTS) \
	$(PYTHON) lmcache-vllm-extended/request_generator_task3.py --n-docs 14
endef

cache-0:   $(RESULTS)
	$(call run_cache_test,cache_0gb)

cache-01:  $(RESULTS)
	$(call run_cache_test,cache_01gb)

cache-05:  $(RESULTS)
	$(call run_cache_test,cache_05gb)

cache-1:   $(RESULTS)
	$(call run_cache_test,cache_1gb)

cache-2:   $(RESULTS)
	$(call run_cache_test,cache_2gb)

cache-4:   $(RESULTS)
	$(call run_cache_test,cache_4gb)

cache-8:   $(RESULTS)
	$(call run_cache_test,cache_8gb)

# ── N-docs experiments (fixed cache config, vary database size) ───────────── 
define run_ndocs_test
	$(call stage_docs,$(1))
	RESULTS_LABEL=ndocs_$(1) RESULTS_DIR=$(RESULTS) \
	$(PYTHON) lmcache-vllm-extended/request_generator_task3.py --n-docs $(1)
endef

ndocs-2:   $(RESULTS)
	$(call run_ndocs_test,2)

ndocs-4:   $(RESULTS)
	$(call run_ndocs_test,4)

ndocs-7:   $(RESULTS)
	$(call run_ndocs_test,7)

ndocs-10:  $(RESULTS)
	$(call run_ndocs_test,10)

ndocs-14:  $(RESULTS)
	$(call run_ndocs_test,14)