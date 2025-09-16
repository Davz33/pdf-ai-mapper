"""
Enhanced text preprocessing module for PDF AI Mapper.
Handles text cleaning, tokenization, and key phrase extraction.
"""

import re
import logging
from collections import Counter
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


class TextPreprocessor:
    """Handles text preprocessing and key phrase extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_stopwords()
    
    def _setup_stopwords(self):
        """Setup enhanced stopwords list."""
        self.stop_words = set(stopwords.words('english'))
        # Add common but less meaningful words
        additional_stopwords = {
            'said', 'says', 'would', 'could', 'should', 'might', 'may', 'must', 'shall', 'will',
            'can', 'cannot', 'couldnt', 'wouldnt', 'shouldnt', 'dont', 'doesnt', 'didnt', 'wont', 'cant',
            'shant', 'aint', 'arent', 'isnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
            'do', 'does', 'did', 'done', 'doing', 'go', 'goes', 'went', 'gone', 'going',
            'get', 'gets', 'got', 'gotten', 'getting', 'come', 'comes', 'came', 'coming',
            'see', 'sees', 'saw', 'seen', 'seeing', 'know', 'knows', 'knew', 'known', 'knowing',
            'think', 'thinks', 'thought', 'thinking', 'make', 'makes', 'made', 'making',
            'take', 'takes', 'took', 'taken', 'taking', 'give', 'gives', 'gave', 'given', 'giving',
            'find', 'finds', 'found', 'finding', 'look', 'looks', 'looked', 'looking',
            'use', 'uses', 'used', 'using', 'work', 'works', 'worked', 'working',
            'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying',
            'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing',
            'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming',
            'leave', 'leaves', 'left', 'leaving', 'put', 'puts', 'putting',
            'tell', 'tells', 'told', 'telling', 'seem', 'seems', 'seemed', 'seeming',
            'let', 'lets', 'letting', 'help', 'helps', 'helped', 'helping',
            'keep', 'keeps', 'kept', 'keeping', 'turn', 'turns', 'turned', 'turning',
            'start', 'starts', 'started', 'starting', 'show', 'shows', 'showed', 'showing',
            'hear', 'hears', 'heard', 'hearing', 'play', 'plays', 'played', 'playing',
            'run', 'runs', 'ran', 'running', 'move', 'moves', 'moved', 'moving',
            'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing',
            'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing',
            'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing',
            'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting',
            'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing',
            'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting',
            'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing',
            'set', 'sets', 'setting', 'learn', 'learns', 'learned', 'learning',
            'change', 'changes', 'changed', 'changing', 'lead', 'leads', 'led', 'leading',
            'understand', 'understands', 'understood', 'understanding',
            'watch', 'watches', 'watched', 'watching', 'follow', 'follows', 'followed', 'following',
            'stop', 'stops', 'stopped', 'stopping', 'create', 'creates', 'created', 'creating',
            'speak', 'speaks', 'spoke', 'spoken', 'speaking', 'read', 'reads', 'reading',
            'allow', 'allows', 'allowed', 'allowing', 'add', 'adds', 'added', 'adding',
            'spend', 'spends', 'spent', 'spending', 'grow', 'grows', 'grew', 'grown', 'growing',
            'open', 'opens', 'opened', 'opening', 'walk', 'walks', 'walked', 'walking',
            'win', 'wins', 'won', 'winning', 'offer', 'offers', 'offered', 'offering',
            'remember', 'remembers', 'remembered', 'remembering', 'love', 'loves', 'loved', 'loving',
            'consider', 'considers', 'considered', 'considering', 'appear', 'appears', 'appeared', 'appearing',
            'buy', 'buys', 'bought', 'buying', 'wait', 'waits', 'waited', 'waiting',
            'serve', 'serves', 'served', 'serving', 'die', 'dies', 'died', 'dying',
            'send', 'sends', 'sent', 'sending', 'expect', 'expects', 'expected', 'expecting',
            'build', 'builds', 'built', 'building', 'stay', 'stays', 'stayed', 'staying',
            'fall', 'falls', 'fell', 'fallen', 'falling', 'cut', 'cuts', 'cutting',
            'reach', 'reaches', 'reached', 'reaching', 'kill', 'kills', 'killed', 'killing',
            'remain', 'remains', 'remained', 'remaining', 'suggest', 'suggests', 'suggested', 'suggesting',
            'raise', 'raises', 'raised', 'raising', 'pass', 'passes', 'passed', 'passing',
            'sell', 'sells', 'sold', 'selling', 'require', 'requires', 'required', 'requiring',
            'report', 'reports', 'reported', 'reporting', 'decide', 'decides', 'decided', 'deciding',
            'pull', 'pulls', 'pulled', 'pulling', 'like', 'ofthe', 'things', 'posterior', 'anterior', 'surface'
        }
        self.stop_words.update(additional_stopwords)
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing that keeps meaningful words and extracts key phrases."""
        # Handle error messages
        if text.startswith("Error:"):
            return text
            
        try:
            # Log the original text length
            self.logger.info(f"Preprocessing text of length {len(text)}")
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove Greek characters and other non-Latin scripts
            text = re.sub(r'[\u0370-\u03FF\u1F00-\u1FFF]', ' ', text)  # Greek characters
            text = re.sub(r'[\u0400-\u04FF]', ' ', text)  # Cyrillic characters
            text = re.sub(r'[\u4E00-\u9FFF]', ' ', text)  # Chinese characters
            text = re.sub(r'[\u0600-\u06FF]', ' ', text)  # Arabic characters
            
            # Remove special characters but keep hyphens for compound words
            text = re.sub(r'[^\w\s-]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Filter tokens based on POS tags and meaningfulness
            meaningful_tokens = []
            for token, pos in pos_tags:
                # Keep nouns, adjectives, and meaningful words
                if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) and \
                   len(token) >= 3 and \
                   token not in self.stop_words and \
                   not token.isdigit():
                    meaningful_tokens.append(token)
            
            # Extract key phrases (bigrams and trigrams)
            key_phrases = self._extract_key_phrases(meaningful_tokens)
            
            # Combine meaningful tokens with key phrases
            all_terms = meaningful_tokens + key_phrases
            
            processed_text = ' '.join(all_terms)
            self.logger.info(f"Preprocessed text length: {len(processed_text)}")
            
            return processed_text
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text
    
    def _extract_key_phrases(self, tokens):
        """Extract meaningful key phrases (bigrams and trigrams)."""
        try:
            # Extract bigrams
            bigrams = list(itertools.combinations(tokens, 2))
            bigram_phrases = [' '.join(bigram) for bigram in bigrams if len(bigram[0]) >= 3 and len(bigram[1]) >= 3]
            
            # Extract trigrams
            trigrams = list(itertools.combinations(tokens, 3))
            trigram_phrases = [' '.join(trigram) for trigram in trigrams if all(len(word) >= 3 for word in trigram)]
            
            # Count frequency and return most common phrases
            all_phrases = bigram_phrases + trigram_phrases
            phrase_counts = Counter(all_phrases)
            
            # Return phrases that appear at least twice
            return [phrase for phrase, count in phrase_counts.most_common(20) if count >= 2]
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}")
            return []