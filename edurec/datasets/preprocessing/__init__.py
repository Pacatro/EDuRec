"""Canonical preprocessing: cleaning, reduction, splits and feature encoders.

Most sources call ``reduction.apply_common_filters`` before passing selected
tables to ``build.build_processed`` for splitting, reduction, labelling and
feature encoding. Sources with a streaming reduction path, such as MOOCCubeX,
apply the same source-level filters while reading their files.
``reduction.reduce_dataset`` samples whole users, reduces the catalog and
optionally prunes concept memberships from static course metadata before text
encoding and graph construction. Model-specific structures belong to the model
or baseline that consumes them.
"""
