{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods_summary %}
   {% if methods %}

   {% for item in ['__new__', '__init__'] %}
   {% if item in methods %}
   {% set dummy = methods.remove(item) %}
   {% endif %}
   {% endfor %}
   {% endif %}

   {% if methods %}
   .. rubric:: {{ _('Methods Summary') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes_summary %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes Summary') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods_documentation %}
   {% if methods %}
   .. rubric:: {{ _('Methods Documentation') }}
   {% for item in methods %}
   .. automethod:: {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes_documentation %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes Documentation') }}
   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}
