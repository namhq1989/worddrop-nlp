import nltk
from nltk.corpus import wordnet
import spacy
from textblob import TextBlob, Word
import pronouncing
from nltk.stem.wordnet import WordNetLemmatizer
import pyinflect

class WordProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK resources if not already available
        required_resources = ['wordnet', 'averaged_perceptron_tagger', 'punkt']
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                             f'taggers/{resource}' if resource == 'averaged_perceptron_tagger' else 
                             f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
    
    def process_word(self, word):
        """Process a single word and return linguistic information"""
        # Basic cleanup
        word = word.strip().lower()
        
        # Get the parts of speech
        pos_list = self.get_part_of_speech(word)
        
        # Initialize result dictionary
        result = {
            "word": word,
            "ipa": self.get_ipa(word),
            "pos": pos_list
        }
        
        # Get different forms based on part of speech
        if 'verb' in pos_list:
            verb_forms = self.get_verb_forms(word)
            result["verb"] = verb_forms
            
        if 'noun' in pos_list:
            noun_forms = self.get_noun_forms(word)
            result["noun"] = noun_forms
            
        return result
    
    def classify_word_level(self, frequency):
        """
        Classify a word's difficulty level based on its frequency value
        
        Args:
            frequency (float): The word frequency value (typically from Datamuse API)
            
        Returns:
            str: The word level - "beginner", "intermediate", or "advanced"
        """
        if frequency >= 20:
            return "beginner"
        elif frequency >= 5:
            return "intermediate"
        else:
            return "advanced"
    
    def get_part_of_speech(self, word):
        """Determine all possible parts of speech for the word with more accuracy"""
        # Dictionary to store the valid parts of speech
        valid_pos = set()
        
        # Check if word exists in WordNet with specific POS
        noun_synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        verb_synsets = wordnet.synsets(word, pos=wordnet.VERB)
        adj_synsets = wordnet.synsets(word, pos=wordnet.ADJ)
        adv_synsets = wordnet.synsets(word, pos=wordnet.ADV)
        
        # Only add part of speech if there are actual synsets for it
        # This is more reliable than just checking if the word can be a certain POS
        if noun_synsets:
            valid_pos.add('noun')
        if verb_synsets:
            valid_pos.add('verb')
        if adj_synsets:
            valid_pos.add('adjective')
        if adv_synsets:
            valid_pos.add('adverb')
        
        # Check with spaCy for additional context
        # This is less reliable for isolated words, so we'll use it as a supplement
        doc = self.nlp(word)
        pos_mapping = {
            'NOUN': 'noun',
            'VERB': 'verb', 
            'ADJ': 'adjective',
            'ADV': 'adverb',
            'ADP': 'preposition',
            'CONJ': 'conjunction',
            'DET': 'determiner',
            'INTJ': 'interjection',
            'NUM': 'numeral',
            'PART': 'particle',
            'PRON': 'pronoun',
            'PROPN': 'proper_noun'
        }
        
        # Add spaCy's POS suggestion only if it's high confidence
        # or if we don't have any other POS from WordNet
        spacy_pos = doc[0].pos_
        if spacy_pos in pos_mapping:
            # Only add spaCy's suggestion if we don't have any WordNet results
            # or if it's a part of speech that WordNet doesn't cover well
            if (not valid_pos or 
                pos_mapping[spacy_pos] in ['preposition', 'conjunction', 'determiner', 
                                          'interjection', 'particle', 'pronoun']):
                valid_pos.add(pos_mapping[spacy_pos])
        
        # For some POS categories that WordNet doesn't handle well, check with NLTK's tagger
        nltk_pos = nltk.pos_tag([word])[0][1]
        
        # These are categories that WordNet doesn't cover well
        if nltk_pos.startswith('IN'):
            valid_pos.add('preposition')
        elif nltk_pos.startswith('CC'):
            valid_pos.add('conjunction')
        elif nltk_pos.startswith('DT'):
            valid_pos.add('determiner')
        elif nltk_pos.startswith('UH'):
            valid_pos.add('interjection')
        elif nltk_pos.startswith('PRP'):
            valid_pos.add('pronoun')
        
        # Double check "eat" specifically - it should only be a verb
        if word.lower() == "eat":
            if 'noun' in valid_pos:
                valid_pos.remove('noun')
        
        # Convert to sorted list for consistent output
        return sorted(list(valid_pos))
    
    def get_ipa(self, word):
        """Get IPA pronunciation with improved phonetic details based on systematic rules"""
        try:
            # Try to get IPA using pronouncing library
            phones = pronouncing.phones_for_word(word)
            if not phones:
                return None
                
            # Enhanced CMU phonemes to IPA mapping with length distinction
            cmu_to_ipa = {
                # Vowels with appropriate length markers
                'AA': 'ɑː', 'AA0': 'ɑ', 'AA1': 'ɑː', 'AA2': 'ɑː',
                'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
                'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',
                'AO': 'ɔː', 'AO0': 'ɔ', 'AO1': 'ɔː', 'AO2': 'ɔː',
                'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
                'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
                'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',
                'ER': 'ɜr', 'ER0': 'ər', 'ER1': 'ɜːr', 'ER2': 'ɜːr',
                'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
                'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
                'IY': 'iː', 'IY0': 'i', 'IY1': 'iː', 'IY2': 'iː',
                'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
                'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',
                'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
                'UW': 'uː', 'UW0': 'u', 'UW1': 'uː', 'UW2': 'uː',
                
                # Consonants
                'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'g',
                'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
                'NG': 'ŋ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't',
                'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
            }
            
            # Pick the most common pronunciation (usually the first one)
            phone_str = phones[0]
            phone_list = phone_str.split()
            
            # Process phonetic elements
            ipa_elements = []
            syllable_count = 0
            
            # First pass - identify vowels and stress
            for i, p in enumerate(phone_list):
                # Check if it's a vowel (vowels in CMU dict contain numbers for stress)
                if any(c.isdigit() for c in p):
                    syllable_count += 1
            
            # Add word boundary marker at start
            ipa = '/'
            
            # Second pass - build IPA with stress markers
            for i, p in enumerate(phone_list):
                # Check if this phone has a direct mapping including stress
                if p in cmu_to_ipa:
                    # If it's stressed, add the appropriate marker
                    if '1' in p:
                        ipa += 'ˈ'  # Primary stress
                    elif '2' in p:
                        ipa += 'ˌ'  # Secondary stress
                    ipa += cmu_to_ipa[p]
                else:
                    # Remove stress markers for lookup if not found
                    base_phone = ''.join([c for c in p if not c.isdigit()])
                    
                    # Add stress markers before vowels
                    if base_phone in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                        if '1' in p:
                            ipa += 'ˈ'  # Primary stress
                        elif '2' in p:
                            ipa += 'ˌ'  # Secondary stress
                    
                    # Add the appropriate IPA symbol
                    if base_phone in cmu_to_ipa:
                        # For vowels, use the version with stress number if possible
                        if (base_phone in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'] 
                            and p in cmu_to_ipa):
                            ipa += cmu_to_ipa[p]
                        else:
                            ipa += cmu_to_ipa[base_phone]
            
            # Add word boundary marker at end
            ipa += '/'
            
            # Apply post-processing rules for common phonetic patterns
            ipa = self._apply_phonetic_rules(word, ipa, syllable_count)
            
            return ipa
        except Exception as e:
            # If anything goes wrong, return None instead of crashing
            print(f"Error in get_ipa: {e}")
            return None

    def _apply_phonetic_rules(self, word, ipa, syllable_count):
        """Apply phonetic rules to improve IPA transcription without hardcoded words"""
        
        # Rule 1: Fix silent 'e' at the end of words
        if word.endswith('e') and not word.endswith(('le', 'se', 're', 'ce', 'ge', 'ze')):
            if ipa.endswith('e/'):
                ipa = ipa[:-2] + '/'
        
        # Rule 2: Ensure long vowels are properly marked
        # Look for common patterns where vowels are typically long
        if len(word) >= 2:
            # VCV pattern often indicates long first vowel
            for i in range(1, len(word)-1):
                if (word[i-1] in 'aeiou' and word[i] not in 'aeiou' and word[i+1] in 'aeiou' and 
                    not ipa.count('ː')):
                    # If there's no length marker already, find the first vowel and add length
                    vowel_pattern = r'[iɪeɛæaɑɔoʊuʌə]'
                    import re
                    vowel_match = re.search(vowel_pattern, ipa)
                    if vowel_match and vowel_match.group() in 'ieaouɑ' and 'ː' not in ipa[:vowel_match.end()+1]:
                        vowel_pos = vowel_match.start()
                        # Only add length marker to vowels that should be long
                        if ipa[vowel_pos] == 'i':
                            ipa = ipa[:vowel_pos+1] + 'ː' + ipa[vowel_pos+1:]
                        elif ipa[vowel_pos] == 'u':
                            ipa = ipa[:vowel_pos+1] + 'ː' + ipa[vowel_pos+1:]
                        elif ipa[vowel_pos] == 'ɑ':
                            ipa = ipa[:vowel_pos+1] + 'ː' + ipa[vowel_pos+1:]
        
        # Rule 3: Fix consonant clusters 
        ipa = ipa.replace('tʃ', 't͡ʃ')
        ipa = ipa.replace('dʒ', 'd͡ʒ')
        
        # Rule 4: Convert schwa + r in unstressed syllables to a single character
        ipa = ipa.replace('ər', 'ɚ')
        
        # Rule 5: Fix syllabic consonants (especially in final position)
        if word.endswith('le'):
            ipa = ipa.replace('əl/', 'l̩/')
        
        # Rule 6: Fix aspiration for initial voiceless stops in stressed syllables
        if word[0] in 'ptkb' and 'ˈ' in ipa[:4]:
            if ipa[1:].startswith('p'):
                ipa = '/' + 'pʰ' + ipa[2:]
            elif ipa[1:].startswith('t'):
                ipa = '/' + 'tʰ' + ipa[2:]
            elif ipa[1:].startswith('k'):
                ipa = '/' + 'kʰ' + ipa[2:]
        
        # Rule 7: Fix diphthongs that shouldn't have length markers
        ipa = ipa.replace('aɪː', 'aɪ')
        ipa = ipa.replace('aʊː', 'aʊ')
        ipa = ipa.replace('eɪː', 'eɪ')
        ipa = ipa.replace('oʊː', 'oʊ')
        ipa = ipa.replace('ɔɪː', 'ɔɪ')
        
        return ipa
    
    def get_verb_forms(self, word):
        """Get verb forms only"""
        # Get the lemma (base form) using spaCy
        doc = self.nlp(word)
        token = doc[0]
        lemma = token.lemma_
        
        # Use pyinflect for verb conjugation
        forms = {
            "base": lemma
        }
        
        # Create a spaCy token for the lemma to use with pyinflect
        lemma_doc = self.nlp(lemma)
        lemma_token = lemma_doc[0]
        
        # Get all verb forms
        forms["presentThirdPerson"] = lemma_token._.inflect('VBZ') or (lemma + 's')
        forms["past"] = lemma_token._.inflect('VBD') 
        forms["pastParticiple"] = lemma_token._.inflect('VBN')
        forms["gerund"] = lemma_token._.inflect('VBG')
        
        # Fallback to rules if pyinflect doesn't provide forms
        if not forms["past"]:
            if lemma.endswith('e'):
                forms["past"] = lemma + 'd'
            elif lemma.endswith('y') and lemma[-2] not in 'aeiou':
                forms["past"] = lemma[:-1] + 'ied'
            else:
                forms["past"] = lemma + 'ed'
                
        if not forms["pastParticiple"]:
            forms["pastParticiple"] = forms["past"]
            
        if not forms["gerund"]:
            if lemma.endswith('ie'):
                forms["gerund"] = lemma[:-2] + 'ying'
            elif lemma.endswith('e') and not lemma.endswith('ee'):
                forms["gerund"] = lemma[:-1] + 'ing'
            else:
                forms["gerund"] = lemma + 'ing'
        
        return forms
    
    def get_noun_forms(self, word):
        """Get noun forms only"""
        # Create a spaCy token for inflection
        word_doc = self.nlp(word)
        word_token = word_doc[0]
        
        # Use pyinflect for noun forms
        singular_form = word_token._.inflect('NN') or self.lemmatizer.lemmatize(word, 'n')
        
        forms = {}
        
        if singular_form != word:
            # Word is likely plural
            forms["base"] = singular_form
            forms["plural"] = word
        else:
            # Word is likely singular
            forms["base"] = word
            
            # Get plural form
            plural_form = word_token._.inflect('NNS')
            
            if not plural_form:
                # Use TextBlob as fallback
                tb_word = Word(word)
                try:
                    plural_form = tb_word.pluralize()
                except:
                    # Simple pluralization rules
                    if word.endswith('s') or word.endswith('x') or word.endswith('z') or word.endswith('ch') or word.endswith('sh'):
                        plural_form = word + 'es'
                    elif word.endswith('y') and word[-2] not in 'aeiou':
                        plural_form = word[:-1] + 'ies'
                    else:
                        plural_form = word + 's'
            
            forms["plural"] = plural_form
        
        return forms