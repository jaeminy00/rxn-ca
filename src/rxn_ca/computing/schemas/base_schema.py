from monty.json import MSONable, MontyDecoder, MontyEncoder
import dataclasses
import json

class BaseSchema(MSONable):

    @classmethod
    def from_file(cls, fname):
        with open(fname, "rb") as f:
            return json.load(f, cls=MontyDecoder)
    
    def to_file(self, fname):
        with open(fname, "w+") as f:
            json.dump(self.as_dict(), f, cls=MontyEncoder)
    
    def as_dict(self) -> dict:
        """Convert to dictionary for MSONable serialization.
        
        For dataclasses, manually builds dict to preserve MSONable metadata in nested objects.
        """
        if dataclasses.is_dataclass(self):
            d = {}
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                # If it's MSONable, call its as_dict(); otherwise use value as-is
                if isinstance(value, MSONable):
                    d[field.name] = value.as_dict()
                elif isinstance(value, list):
                    d[field.name] = [item.as_dict() if isinstance(item, MSONable) else item for item in value]
                elif isinstance(value, dict):
                    d[field.name] = {k: v.as_dict() if isinstance(v, MSONable) else v for k, v in value.items()}
                else:
                    d[field.name] = value
            d["@module"] = type(self).__module__
            d["@class"] = type(self).__name__
            return d
        else:
            # Use MSONable's default implementation
            return super().as_dict()
    