simulator: simulator.py
	@cp simulator.py simulator
	@chmod +x simulator

.PHONY: clean
clean:
	@rm -f simulator
