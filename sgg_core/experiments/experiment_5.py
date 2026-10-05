"""Experiment V: validation-gated object-head mitigation.

The paper protocol evaluates one preregistered TDE-Motifs family under matched
supervised-control and grounding modes with three fine-tuning seeds.
"""

from sgg_core.mitigation.run_mitigation import main


if __name__ == "__main__":
    main()
