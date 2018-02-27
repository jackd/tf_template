def register_families():
    from tf_template.model.mats.register import register_all
    from tf_template.model.example_builder import register_example_family
    register_all()
    register_example_family()
