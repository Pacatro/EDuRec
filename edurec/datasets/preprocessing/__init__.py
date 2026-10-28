"""Canonical preprocessing: cleaning, splits and fitted feature encoders.

Sources use ``prepare.apply_common_filters`` before passing selected tables to
``build.build_processed`` for splitting, labelling and feature encoding.
Model-specific structures belong to the model or baseline that consumes them.
"""
