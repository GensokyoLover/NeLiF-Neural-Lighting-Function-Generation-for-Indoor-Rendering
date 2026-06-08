class Serializable():
    def serialize(self, *args, **kwargs):
        return { attr_name: attr_val for (attr_name, attr_val) in self.__dict__.items() if attr_name[0] != '_' }

    def serialize_all(self, *args, **kwargs):
        return { attr_name: attr_val for (attr_name, attr_val) in self.__dict__.items() }

from typing import Sequence, Mapping

# class Serializable():
#     def _serialize_attribute(self, value, **kwargs):
#         if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
#             return [self._serialize_attribute(e) for e in value]
#         elif isinstance(value, Mapping):
#             return {k: self._serialize_attribute(v) for k, v in value.items()}
#         elif isinstance(value, Serializable):
#             return value.serialize(**kwargs)
#         else:
#             return value
    
#     def serialize(self, **kwargs):
#         result = {
#             prop_name: self._serialize_attribute(prop, **kwargs) 
#             for prop_name in dir(self) 
#                 if prop_name[0] != '_' and 
#                 prop_name not in kwargs.get('hidden_properties', {}) and
#                 not callable(prop := getattr(self, prop_name)) 
#             }
#         if hasattr(self, 'has_name') and not self.has_name():
#             result.pop('name')
#         return result
