import requests
import logging

class DatamuseAPI:
    def __init__(self):
        """Initialize the Datamuse API client"""
        self.base_url = "https://api.datamuse.com/words"
        self.logger = logging.getLogger(__name__)
        
        # POS mapping from Datamuse format to our application format
        self.pos_mapping = {
            "adj": "adj",     # Adjective
            "adv": "adv",     # Adverb
            "n": "noun",      # Noun
            "v": "verb",      # Verb
            "u": "intj",      # Interjection
            "c": "conj",      # Conjunction
            "p": "adp",       # Preposition
            "r": "pron",      # Pronoun
            "d": "det",       # Determiner
            "i": "part",      # Particle
            "num": "num",     # Numeral
            "pron": "pron",   # Pronoun
            "prep": "adp",    # Preposition (alternative notation)
            "conj": "conj",   # Conjunction (alternative notation)
            "interj": "intj", # Interjection (alternative notation)
            "proper noun": "propn", # Proper noun
            "aux": "aux",     # Auxiliary
            "propn": "propn", # Proper noun
            "sconj": "sconj", # Subordinating conjunction
            "sym": "sym",     # Symbol
            "punct": "punct", # Punctuation
            "x": "x"          # Other
        }
    
    def search_term(self, term):
        """
        Search for word information using Datamuse API
        Returns processed information about the word
        """
        result = {
            "definitions": [],
            "frequency": 0.0,
            "ipa": ""
        }
        
        try:
            params = {
                "sp": term,
                "qe": "sp",
                "md": "drf",
                "ipa": "1",
                "max": "5"  # Get more results to improve matching
            }
            
            self.logger.info(f"Making Datamuse API request for term: '{term}'")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            api_results = response.json()
            
            if not api_results:
                self.logger.warning(f"Datamuse API search result is empty for term: '{term}'")
                return result
            
            # Find the best matching result (prioritize exact match, then case-insensitive)
            best_match = None
            for item in api_results:
                if item["word"] == term:  # Exact match
                    best_match = item
                    break
                elif item["word"].lower() == term.lower() and best_match is None:  # Case-insensitive match
                    best_match = item
            
            # If no exact match found, use the first result
            if best_match is None and api_results:
                best_match = api_results[0]
                self.logger.info(f"Using closest match: '{best_match['word']}' for term: '{term}'")
            
            if best_match is None:
                return result
            
            # Process definitions if available
            if "defs" in best_match and best_match["defs"]:
                for definition in best_match["defs"]:
                    parts = definition.split("\t")
                    if len(parts) == 2:
                        original_pos = parts[0]
                        definition_text = parts[1].strip()
                        definition_text = definition_text.rstrip(".")
                        definition_text = self._uncapitalize_definition(definition_text)
                        
                        # Map the POS to our format
                        mapped_pos = self._map_pos(original_pos)
                        
                        result["definitions"].append({
                            "pos": mapped_pos,
                            "definition": definition_text
                        })
            
            # Process tags if available
            if "tags" in best_match:
                for tag in best_match["tags"]:
                    if tag.startswith("ipa_pron:"):
                        result["ipa"] = tag.split(":", 1)[1]
                    elif tag.startswith("f:"):
                        try:
                            result["frequency"] = float(tag.split(":", 1)[1])
                        except ValueError:
                            pass
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error searching term with Datamuse: {str(e)}", extra={"term": term})
            return result
    
    def _map_pos(self, datamuse_pos):
        """Map Datamuse POS to our application format"""
        # Remove any qualifiers in parentheses
        clean_pos = datamuse_pos.split('(')[0].strip()
        
        # Check if it's already in our format
        if clean_pos in self.pos_mapping.values():
            return clean_pos
        
        # Map to our format or return 'x' (other) if not found
        return self.pos_mapping.get(clean_pos, "x")
    
    def _uncapitalize_definition(self, s):
        """Uncapitalize the first letter of definition, preserving annotations"""
        s = s.strip()
        s = s.rstrip(".")
        
        # Find position of the first character after annotation
        closing_paren_index = s.rfind(")")
        if closing_paren_index != -1 and closing_paren_index < len(s) - 1:
            annotation = s[:closing_paren_index + 1]
            definition = s[closing_paren_index + 1:].strip()
            if definition:
                return f"{annotation} {definition[0].lower()}{definition[1:]}"
        else:
            # No annotation, uncapitalize the whole string
            definition = s.strip()
            if definition:
                return f"{definition[0].lower()}{definition[1:]}"
        
        return s  # If the string is empty, return it as is