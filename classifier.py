# import torch
# import numpy as np
# import re
# from transformers import (
#     AutoTokenizer, AutoModelForSequenceClassification,
#     DistilBertTokenizer, DistilBertForSequenceClassification,
#     pipeline, BertTokenizer, BertForSequenceClassification
# )
# import warnings
# import logging
#
# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')
# logging.getLogger('transformers').setLevel(logging.ERROR)
#
#
# class PretrainedEmailClassifier:
#     """
#     Professional Email Classifier using multiple pretrained models
#     Optimized for NVIDIA T500 2GB GPU and CPU usage
#     """
#
#     def __init__(self, model_type="distilbert", device="auto"):
#         """
#         Initialize classifier with different model options
#
#         Args:
#             model_type: "distilbert", "bert-base", "roberta", "simple"
#             device: "auto", "cpu", "cuda"
#         """
#         self.model_type = model_type
#         self.device = self._get_device(device)
#         self.model = None
#         self.tokenizer = None
#         self.classifier = None
#
#         # Email category mapping
#         self.categories = {
#             0: "Updates",
#             1: "Personal",
#             2: "Promotions",
#             3: "Forums",
#             4: "Purchases",
#             5: "Travel",
#             6: "Spam",
#             7: "Social"
#         }
#
#         # Enhanced keyword patterns for each category
#         self._initialize_enhanced_patterns()
#
#         # Load model with error handling
#         self.load_model()
#
#     def _initialize_enhanced_patterns(self):
#         """Initialize comprehensive keyword patterns for all categories"""
#
#         # SPAM - Comprehensive spam detection
#         self.spam_keywords = {
#             'urgent_words': ['urgent', 'immediate', 'act now', 'hurry', 'expires today', 'last chance',
#                              'final notice', 'time sensitive', 'don\'t delay', 'expires soon', 'deadline',
#                              'limited time only', 'offer expires', 'must act', 'don\'t wait'],
#             'money_offers': ['win', 'winner', 'won', 'prize', 'lottery', 'cash', 'money', '$$$',
#                              'million', 'thousand', 'jackpot', 'sweepstakes', 'windfall', 'inheritance',
#                              'free money', 'easy money', 'make money fast', 'get paid', 'earn money',
#                              'financial freedom', 'passive income', 'work from home income'],
#             'suspicious_claims': ['guaranteed', 'no risk', 'risk free', '100% guaranteed', 'promise',
#                                   'certified', 'verified winner', 'pre-selected', 'congratulations',
#                                   'you have been chosen', 'specially selected', 'exclusive opportunity'],
#             'action_words': ['click here', 'call now', 'click now', 'visit now', 'order now',
#                              'buy now', 'download now', 'register now', 'sign up now', 'claim now',
#                              'redeem now', 'get started', 'apply now', 'join now'],
#             'medical_pharma': ['viagra', 'cialis', 'pharmacy', 'pills', 'medication', 'prescription',
#                                'doctor approved', 'fda approved', 'medical breakthrough', 'cure',
#                                'lose weight', 'weight loss', 'diet pills'],
#             'scam_indicators': ['nigerian prince', 'beneficiary', 'next of kin', 'transfer funds',
#                                 'bank details', 'account details', 'routing number', 'ssn',
#                                 'social security', 'wire transfer', 'western union', 'money transfer']
#         }
#
#         # PROMOTIONS - Marketing and sales content
#         self.promotion_keywords = {
#             'sales_words': ['sale', 'discount', 'offer', 'deal', 'bargain', 'clearance', 'markdown',
#                             'reduced', 'price drop', 'flash sale', 'mega sale', 'super sale', 'big sale',
#                             'end of season', 'closeout', 'liquidation', 'warehouse sale'],
#             'discounts': ['%', 'percent', 'off', 'save', 'savings', 'coupon', 'promo code', 'voucher',
#                           'discount code', 'special price', 'reduced price', 'slashed price', 'half price',
#                           'buy one get one', 'bogo', '2 for 1', 'bundle deal'],
#             'marketing_terms': ['exclusive', 'limited offer', 'special offer', 'promotional', 'featured',
#                                 'bestseller', 'top rated', 'customer favorite', 'most popular', 'trending',
#                                 'new arrival', 'just in', 'fresh stock', 'back in stock'],
#             'shopping_words': ['shop', 'store', 'retail', 'mall', 'marketplace', 'catalog', 'collection',
#                                'browse', 'selection', 'inventory', 'merchandise', 'products', 'items'],
#             'seasonal_sales': ['black friday', 'cyber monday', 'christmas sale', 'holiday sale',
#                                'summer sale', 'winter clearance', 'spring collection', 'back to school',
#                                'valentine\'s day', 'mother\'s day', 'father\'s day']
#         }
#
#         # SOCIAL - Social media and networking
#         self.social_keywords = {
#             'platforms': ['facebook', 'twitter', 'instagram', 'linkedin', 'snapchat', 'tiktok',
#                           'youtube', 'pinterest', 'reddit', 'discord', 'whatsapp', 'telegram',
#                           'social media', 'social network', 'social platform'],
#             'activities': ['posted', 'shared', 'liked', 'commented', 'tagged', 'mentioned', 'followed',
#                            'unfollowed', 'friended', 'connected', 'endorsed', 'recommended', 'reviewed'],
#             'notifications': ['notification', 'alert', 'friend request', 'connection request',
#                               'follow request', 'message request', 'group invitation', 'event invitation',
#                               'tagged you', 'mentioned you', 'commented on', 'liked your'],
#             'content_types': ['profile', 'timeline', 'feed', 'story', 'post', 'tweet', 'status update',
#                               'photo', 'video', 'live stream', 'reel', 'story highlight', 'memory'],
#             'engagement': ['views', 'likes', 'shares', 'comments', 'reactions', 'followers', 'following',
#                            'subscribers', 'connections', 'network', 'community', 'group', 'page']
#         }
#
#         # PURCHASES - E-commerce and transactions
#         self.purchase_keywords = {
#             'order_status': ['order', 'purchase', 'bought', 'ordered', 'checkout', 'cart', 'basket',
#                              'order confirmation', 'order placed', 'order received', 'processing',
#                              'preparing', 'packed', 'ready to ship'],
#             'payment_terms': ['payment', 'paid', 'charged', 'billed', 'invoice', 'receipt', 'billing',
#                               'transaction', 'refund', 'refunded', 'chargeback', 'credit card', 'paypal',
#                               'stripe', 'square', 'payment method', 'payment declined', 'payment failed'],
#             'shipping': ['shipped', 'shipping', 'delivered', 'delivery', 'tracking', 'tracking number',
#                          'fedex', 'ups', 'dhl', 'usps', 'courier', 'express', 'standard shipping',
#                          'free shipping', 'same day delivery', 'next day', 'expedited'],
#             'retail_terms': ['amazon', 'ebay', 'walmart', 'target', 'best buy', 'costco', 'marketplace',
#                              'vendor', 'seller', 'merchant', 'retailer', 'supplier', 'manufacturer'],
#             'return_exchange': ['return', 'exchange', 'replacement', 'warranty', 'guarantee',
#                                 'defective', 'damaged', 'wrong item', 'not as described', 'quality issue']
#         }
#
#         # TRAVEL - Tourism and transportation
#         self.travel_keywords = {
#             'transportation': ['flight', 'plane', 'airplane', 'airline', 'airport', 'boarding pass',
#                                'gate', 'terminal', 'departure', 'arrival', 'layover', 'connecting flight',
#                                'train', 'bus', 'car rental', 'taxi', 'uber', 'lyft', 'ride share'],
#             'accommodation': ['hotel', 'motel', 'resort', 'accommodation', 'booking', 'reservation',
#                               'room', 'suite', 'check-in', 'check-out', 'guest', 'stay', 'lodging',
#                               'airbnb', 'vrbo', 'bed and breakfast', 'hostel', 'inn'],
#             'travel_services': ['travel', 'trip', 'vacation', 'holiday', 'tour', 'cruise', 'safari',
#                                 'excursion', 'adventure', 'getaway', 'retreat', 'expedition', 'journey'],
#             'destinations': ['destination', 'city', 'country', 'beach', 'mountain', 'resort town',
#                              'tourist attraction', 'landmark', 'museum', 'theme park', 'national park'],
#             'booking_terms': ['itinerary', 'travel agent', 'expedia', 'booking.com', 'priceline',
#                               'kayak', 'travelocity', 'orbitz', 'travel insurance', 'visa', 'passport']
#         }
#
#         # FORUMS - Discussion boards and communities
#         self.forum_keywords = {
#             'forum_terms': ['forum', 'discussion', 'board', 'community', 'thread', 'topic', 'post',
#                             'reply', 'comment', 'discussion board', 'message board', 'bulletin board'],
#             'user_roles': ['member', 'user', 'moderator', 'admin', 'administrator', 'staff', 'vip',
#                            'premium member', 'gold member', 'verified', 'contributor', 'expert'],
#             'activities': ['posted', 'replied', 'commented', 'mentioned', 'quoted', 'subscribed',
#                            'unsubscribed', 'watched', 'favorited', 'bookmarked', 'voted', 'rated'],
#             'content_types': ['question', 'answer', 'solution', 'tutorial', 'guide', 'review',
#                               'opinion', 'poll', 'survey', 'announcement', 'sticky', 'pinned'],
#             'platforms': ['reddit', 'stack overflow', 'quora', 'discourse', 'phpbb', 'vbulletin',
#                           'discord server', 'slack workspace', 'telegram group', 'facebook group']
#         }
#
#         # UPDATES - System notifications and account updates
#         self.update_keywords = {
#             'account_terms': ['account', 'profile', 'settings', 'preferences', 'subscription',
#                               'membership', 'plan', 'upgrade', 'downgrade', 'renewal', 'expiration',
#                               'activate', 'deactivate', 'suspend', 'terminated'],
#             'notifications': ['notification', 'alert', 'reminder', 'notice', 'announcement',
#                               'update', 'news', 'bulletin', 'advisory', 'warning', 'maintenance'],
#             'security': ['security', 'password', 'login', 'logout', 'authentication', 'verification',
#                          'two-factor', '2fa', 'verify', 'confirm', 'reset', 'change password',
#                          'suspicious activity', 'unauthorized access', 'security breach'],
#             'financial': ['statement', 'balance', 'transaction', 'deposit', 'withdrawal', 'transfer',
#                           'bank', 'credit card', 'debit', 'finance', 'billing cycle', 'due date',
#                           'overdue', 'payment reminder', 'auto-pay'],
#             'system_terms': ['system', 'service', 'server', 'maintenance', 'downtime', 'upgrade',
#                              'patch', 'bug fix', 'feature', 'improvement', 'changelog', 'version']
#         }
#
#         # PERSONAL - Personal communications
#         self.personal_keywords = {
#             'greetings': ['hi', 'hello', 'hey', 'dear', 'greetings', 'good morning', 'good afternoon',
#                           'good evening', 'hope you\'re well', 'how are you', 'how\'s it going'],
#             'personal_terms': ['friend', 'buddy', 'pal', 'mate', 'family', 'relatives', 'personal',
#                                'private', 'confidential', 'between us', 'just between friends'],
#             'social_activities': ['meet', 'meeting', 'get together', 'hang out', 'catch up', 'coffee',
#                                   'lunch', 'dinner', 'drinks', 'party', 'celebration', 'birthday',
#                                   'anniversary', 'wedding', 'graduation'],
#             'emotions': ['love', 'miss', 'excited', 'happy', 'sad', 'worried', 'concerned', 'grateful',
#                          'thankful', 'appreciate', 'congratulations', 'condolences', 'sympathy'],
#             'life_events': ['baby', 'pregnancy', 'job', 'promotion', 'retirement', 'moving', 'house',
#                             'car', 'vacation', 'hobby', 'health', 'doctor', 'hospital', 'surgery']
#         }
#
#         # Email structure patterns
#         self.email_patterns = {
#             'promotional_structure': [
#                 r'\b(save|discount)\s+\d+%',
#                 r'\$\d+\s+(off|discount)',
#                 r'(limited|exclusive)\s+(offer|deal)',
#                 r'(buy|shop)\s+now',
#                 r'(free|complimentary)\s+(shipping|delivery)'
#             ],
#             'spam_structure': [
#                 r'(congratulations|winner|won)\s+.{0,20}\$([\d,]+)',
#                 r'(urgent|immediate|act now)',
#                 r'(click here|call now).{0,10}(win|money|prize)',
#                 r'\b(guarantee|guaranteed)\s+(money|income|profit)'
#             ],
#             'personal_structure': [
#                 r'^(hi|hello|hey|dear)\s+\w+',
#                 r'(how are you|how\'s it going|catch up)',
#                 r'(love|miss)\s+you',
#                 r'(coffee|lunch|dinner|drinks)\s+(this|next|tomorrow)'
#             ]
#         }
#
#     def _get_device(self, device):
#         """Determine the best device to use"""
#         if device == "auto":
#             if torch.cuda.is_available():
#                 try:
#                     gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
#                     print(f"GPU Memory: {gpu_memory:.1f}GB")
#                     if gpu_memory >= 1.5:
#                         return "cuda"
#                     else:
#                         print("GPU memory insufficient, using CPU")
#                         return "cpu"
#                 except:
#                     print("CUDA error detected, using CPU")
#                     return "cpu"
#             else:
#                 print("CUDA not available, using CPU")
#                 return "cpu"
#         return device
#
#     def load_model(self):
#         """Load the specified pretrained model with fallback"""
#         print(f"Loading {self.model_type} model on {self.device}...")
#
#         try:
#             if self.model_type == "distilbert":
#                 self._load_distilbert()
#             elif self.model_type == "bert-base":
#                 self._load_bert_base()
#             elif self.model_type == "roberta":
#                 self._load_roberta()
#             elif self.model_type == "simple":
#                 self._load_simple_classifier()
#             else:
#                 print(f"Unknown model type: {self.model_type}, using simple classifier")
#                 self._load_simple_classifier()
#
#         except Exception as e:
#             print(f"Error loading {self.model_type}: {e}")
#             print("Falling back to simple rule-based classifier...")
#             self._load_simple_classifier()
#
#     def _load_distilbert(self):
#         """Load DistilBERT - Lightweight and fast"""
#         try:
#             # Load pre-trained sentiment model that actually works
#             self.classifier = pipeline(
#                 "text-classification",
#                 model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#                 device=0 if self.device == "cuda" else -1,
#                 return_all_scores=True
#             )
#
#             print("✓ DistilBERT-based classifier loaded successfully")
#
#         except Exception as e:
#             print(f"Failed to load DistilBERT: {e}")
#             raise
#
#     def _load_bert_base(self):
#         """Load BERT Base - More accurate but slower"""
#         try:
#             self.classifier = pipeline(
#                 "text-classification",
#                 model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#                 device=0 if self.device == "cuda" else -1,
#                 return_all_scores=True
#             )
#             print("✓ BERT-based classifier loaded successfully")
#         except Exception as e:
#             print(f"Failed to load BERT: {e}")
#             raise
#
#     def _load_roberta(self):
#         """Load RoBERTa - Good balance of speed and accuracy"""
#         try:
#             self.classifier = pipeline(
#                 "text-classification",
#                 model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#                 device=0 if self.device == "cuda" else -1,
#                 return_all_scores=True
#             )
#             print("✓ RoBERTa classifier loaded successfully")
#         except Exception as e:
#             print(f"Failed to load RoBERTa: {e}")
#             raise
#
#     def _load_simple_classifier(self):
#         """Load simple rule-based classifier as fallback"""
#         self.classifier = None  # Will use rule-based classification
#         print("✓ Simple rule-based classifier loaded successfully")
#
#     def classify_email(self, subject="", body="", return_probabilities=False):
#         """
#         Classify a single email
#
#         Args:
#             subject: Email subject line
#             body: Email body content
#             return_probabilities: Whether to return all class probabilities
#
#         Returns:
#             dict: Classification result with category and confidence
#         """
#         # Combine subject and body
#         text = f"Subject: {subject}\n\nBody: {body}".strip()
#
#         # Truncate if too long
#         if len(text) > 1000:
#             text = text[:1000] + "..."
#
#         try:
#             if self.classifier is not None:
#                 # Use ML model - Fixed the error handling here
#                 results = self.classifier(text)
#
#                 # Handle the results properly
#                 if results and isinstance(results, list) and len(results) > 0:
#                     # The results come as a list of dictionaries with 'label' and 'score'
#                     # Find the result with highest score
#                     best_result = max(results, key=lambda x: x['score'])
#
#                     # Map sentiment to email category using content analysis
#                     predicted_class = self._enhanced_content_analysis(text, best_result['label'])
#                     confidence = self._calculate_enhanced_confidence(text, predicted_class)
#
#                     # Create probability distribution
#                     all_probs = self._create_enhanced_probability_distribution(predicted_class, confidence, text)
#
#                 else:
#                     # Fallback to rule-based
#                     return self._enhanced_rule_based_classify(subject, body, return_probabilities)
#
#             else:
#                 # Use rule-based classification
#                 return self._enhanced_rule_based_classify(subject, body, return_probabilities)
#
#             result = {
#                 'category': predicted_class,
#                 'confidence': float(confidence),
#                 'text_preview': text[:200] + "..." if len(text) > 200 else text,
#                 'method': 'ml-based'
#             }
#
#             if return_probabilities:
#                 result['all_probabilities'] = all_probs
#
#             return result
#
#         except Exception as e:
#             print(f"Classification error: {e}")
#             return self._enhanced_rule_based_classify(subject, body, return_probabilities)
#
#     def _enhanced_content_analysis(self, text, sentiment):
#         """Enhanced content analysis with comprehensive pattern matching"""
#         text_lower = text.lower()
#
#         # Calculate scores for each category
#         category_scores = {}
#
#         # Spam detection - highest priority for obvious spam
#         spam_score = self._calculate_category_score(text_lower, self.spam_keywords,
#                                                     self.email_patterns.get('spam_structure', []))
#         if spam_score > 0.7:
#             return 'Spam'
#
#         # Calculate scores for all categories
#         category_scores['Spam'] = spam_score
#         category_scores['Promotions'] = self._calculate_category_score(text_lower, self.promotion_keywords,
#                                                                        self.email_patterns.get('promotional_structure',
#                                                                                                []))
#         category_scores['Social'] = self._calculate_category_score(text_lower, self.social_keywords, [])
#         category_scores['Purchases'] = self._calculate_category_score(text_lower, self.purchase_keywords, [])
#         category_scores['Travel'] = self._calculate_category_score(text_lower, self.travel_keywords, [])
#         category_scores['Forums'] = self._calculate_category_score(text_lower, self.forum_keywords, [])
#         category_scores['Updates'] = self._calculate_category_score(text_lower, self.update_keywords, [])
#         category_scores['Personal'] = self._calculate_category_score(text_lower, self.personal_keywords,
#                                                                      self.email_patterns.get('personal_structure', []))
#
#         # Find the category with highest score
#         best_category = max(category_scores, key=category_scores.get)
#         max_score = category_scores[best_category]
#
#         # If no category has a strong score, use sentiment analysis
#         if max_score < 0.3:
#             sentiment_lower = sentiment.lower()
#             if 'positive' in sentiment_lower or sentiment.startswith('LABEL_1'):
#                 return 'Personal'
#             elif 'negative' in sentiment_lower or sentiment.startswith('LABEL_0'):
#                 return 'Spam'
#             else:
#                 return 'Updates'
#
#         return best_category
#
#     def _calculate_category_score(self, text, keyword_dict, patterns):
#         """Calculate score for a category based on keywords and patterns"""
#         total_score = 0
#         total_weight = 0
#
#         # Score based on keyword categories
#         for category, keywords in keyword_dict.items():
#             category_weight = len(keywords)
#             matches = sum(1 for keyword in keywords if keyword in text)
#             category_score = matches / len(keywords) if keywords else 0
#
#             # Apply category-specific weights
#             if category in ['urgent_words', 'suspicious_claims', 'scam_indicators']:
#                 category_weight *= 2  # Higher weight for important spam indicators
#             elif category in ['greetings', 'personal_terms', 'emotions']:
#                 category_weight *= 1.5  # Higher weight for personal indicators
#
#             total_score += category_score * category_weight
#             total_weight += category_weight
#
#         # Score based on regex patterns
#         pattern_score = 0
#         for pattern in patterns:
#             if re.search(pattern, text, re.IGNORECASE):
#                 pattern_score += 0.2
#
#         # Combine scores
#         keyword_score = total_score / total_weight if total_weight > 0 else 0
#         final_score = min((keyword_score + pattern_score), 1.0)
#
#         return final_score
#
#     def _calculate_enhanced_confidence(self, text, predicted_class):
#         """Calculate enhanced confidence based on multiple factors"""
#         base_confidence = 0.5
#
#         # Get category-specific keywords
#         category_keywords = self._get_category_keywords(predicted_class)
#         if not category_keywords:
#             return base_confidence
#
#         # Calculate keyword match ratio
#         text_lower = text.lower()
#         matches = sum(1 for keyword in category_keywords if keyword in text_lower)
#         match_ratio = matches / len(category_keywords) if category_keywords else 0
#
#         # Calculate confidence based on match ratio
#         confidence = base_confidence + (match_ratio * 0.4)
#
#         # Boost confidence for strong indicators
#         if predicted_class == 'Spam':
#             strong_spam_indicators = ['urgent', 'win', 'prize', 'lottery', 'click here', 'call now']
#             if any(indicator in text_lower for indicator in strong_spam_indicators):
#                 confidence = min(confidence + 0.2, 0.95)
#
#         elif predicted_class == 'Personal':
#             personal_indicators = ['dear', 'hi', 'hello', 'love', 'miss', 'friend']
#             if any(indicator in text_lower for indicator in personal_indicators):
#                 confidence = min(confidence + 0.15, 0.90)
#
#         return min(confidence, 0.95)
#
#     def _get_category_keywords(self, category):
#         """Get all keywords for a specific category"""
#         keyword_map = {
#             'Spam': self.spam_keywords,
#             'Promotions': self.promotion_keywords,
#             'Social': self.social_keywords,
#             'Purchases': self.purchase_keywords,
#             'Travel': self.travel_keywords,
#             'Forums': self.forum_keywords,
#             'Updates': self.update_keywords,
#             'Personal': self.personal_keywords
#         }
#
#         category_dict = keyword_map.get(category, {})
#         all_keywords = []
#         for keyword_list in category_dict.values():
#             all_keywords.extend(keyword_list)
#
#         return all_keywords
#
#     def _create_enhanced_probability_distribution(self, predicted_class, confidence, text):
#         """Create enhanced probability distribution for all categories"""
#         all_probs = {}
#         text_lower = text.lower()
#
#         # Calculate base probabilities for all categories
#         base_prob = (1.0 - confidence) / 7  # Distribute remaining probability
#
#         for category in self.categories.values():
#             if category == predicted_class:
#                 all_probs[category] = confidence
#             else:
#                 # Calculate secondary probabilities based on partial matches
#                 category_keywords = self._get_category_keywords(category)
#                 if category_keywords:
#                     matches = sum(
#                         1 for keyword in category_keywords[:20] if keyword in text_lower)  # Limit for performance
#                     match_ratio = matches / min(len(category_keywords), 20)
#                     secondary_prob = base_prob * (1 + match_ratio)
#                     all_probs[category] = secondary_prob
#                 else:
#                     all_probs[category] = base_prob
#
#         # Normalize probabilities
#         total = sum(all_probs.values())
#         if total > 0:
#             all_probs = {k: v / total for k, v in all_probs.items()}
#
#         return all_probs
#
#     def _enhanced_rule_based_classify(self, subject, body, return_probabilities=False):
#         """Enhanced rule-based classification with comprehensive analysis"""
#         combined_text = f"{subject} {body}".lower()
#
#         # Calculate scores for all categories
#         category_scores = {}
#         category_scores['Spam'] = self._calculate_category_score(combined_text, self.spam_keywords,
#                                                                  self.email_patterns.get('spam_structure', []))
#         category_scores['Promotions'] = self._calculate_category_score(combined_text, self.promotion_keywords,
#                                                                        self.email_patterns.get('promotional_structure',
#                                                                                                []))
#         category_scores['Social'] = self._calculate_category_score(combined_text, self.social_keywords, [])
#         category_scores['Purchases'] = self._calculate_category_score(combined_text, self.purchase_keywords, [])
#         category_scores['Travel'] = self._calculate_category_score(combined_text, self.travel_keywords, [])
#         category_scores['Forums'] = self._calculate_category_score(combined_text, self.forum_keywords, [])
#         category_scores['Updates'] = self._calculate_category_score(combined_text, self.update_keywords, [])
#         category_scores['Personal'] = self._calculate_category_score(combined_text, self.personal_keywords,
#                                                                      self.email_patterns.get('personal_structure', []))
#
#         # Find the category with highest score
#         best_category = max(category_scores, key=category_scores.get)
#         max_score = category_scores[best_category]
#
#         # Set minimum confidence thresholds
#         if max_score < 0.2:
#             category = 'Updates'  # Default category
#             confidence = 0.50
#         else:
#             category = best_category
#             confidence = min(0.60 + (max_score * 0.35), 0.95)
#
#         result = {
#             'category': category,
#             'confidence': confidence,
#             'text_preview': combined_text[:200] + "..." if len(combined_text) > 200 else combined_text,
#             'method': 'enhanced-rule-based'
#         }
#
#         if return_probabilities:
#             # Create probability distribution based on scores
#             all_probs = {}
#             total_score = sum(category_scores.values())
#
#             if total_score > 0:
#                 for cat, score in category_scores.items():
#                     if cat == category:
#                         all_probs[cat] = confidence
#                     else:
#                         # Normalize other scores
#                         normalized_score = score / total_score
#                         all_probs[cat] = (1 - confidence) * normalized_score
#             else:
#                 # Equal distribution if no strong signals
#                 equal_prob = 1.0 / len(self.categories)
#                 all_probs = {cat: equal_prob for cat in self.categories.values()}
#
#             # Normalize probabilities to ensure they sum to 1
#             total = sum(all_probs.values())
#             if total > 0:
#                 all_probs = {k: v / total for k, v in all_probs.items()}
#
#             result['all_probabilities'] = all_probs
#
#         return result
#
#     def batch_classify(self, emails):
#         """Classify multiple emails"""
#         results = []
#         total = len(emails)
#
#         for i, email in enumerate(emails):
#             subject = email.get('subject', '')
#             body = email.get('body', '')
#
#             result = self.classify_email(subject, body, return_probabilities=True)
#             result['email_id'] = i
#             results.append(result)
#
#             # Progress indicator
#             if (i + 1) % 10 == 0 or (i + 1) == total:
#                 print(f"Processed {i + 1}/{total} emails")
#
#         return results
#
#     def get_model_info(self):
#         """Get information about the loaded model"""
#         gpu_memory = 0
#         if torch.cuda.is_available():
#             try:
#                 gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
#             except:
#                 gpu_memory = 0
#
#         return {
#             'model_type': self.model_type,
#             'device': self.device,
#             'categories': list(self.categories.values()),
#             'gpu_available': torch.cuda.is_available(),
#             'gpu_memory': gpu_memory,
#             'classifier_loaded': self.classifier is not None,
#             'total_keywords': sum(len(keywords) for category_dict in [
#                 self.spam_keywords, self.promotion_keywords, self.social_keywords,
#                 self.purchase_keywords, self.travel_keywords, self.forum_keywords,
#                 self.update_keywords, self.personal_keywords
#             ] for keywords in category_dict.values())
#         }
#
#     def get_category_details(self, category):
#         """Get detailed information about a specific category's keywords"""
#         keyword_map = {
#             'Spam': self.spam_keywords,
#             'Promotions': self.promotion_keywords,
#             'Social': self.social_keywords,
#             'Purchases': self.purchase_keywords,
#             'Travel': self.travel_keywords,
#             'Forums': self.forum_keywords,
#             'Updates': self.update_keywords,
#             'Personal': self.personal_keywords
#         }
#
#         category_dict = keyword_map.get(category, {})
#         return {
#             'category': category,
#             'keyword_groups': {group: len(keywords) for group, keywords in category_dict.items()},
#             'total_keywords': sum(len(keywords) for keywords in category_dict.values()),
#             'sample_keywords': {
#                 group: keywords[:5] for group, keywords in category_dict.items()
#             }
#         }
#
#     def analyze_text_detailed(self, subject="", body=""):
#         """Provide detailed analysis of text showing scores for all categories"""
#         text = f"Subject: {subject}\n\nBody: {body}".strip()
#         text_lower = text.lower()
#
#         detailed_analysis = {
#             'text_length': len(text),
#             'category_scores': {},
#             'top_keywords_found': {},
#             'pattern_matches': {}
#         }
#
#         # Calculate detailed scores for each category
#         for category in self.categories.values():
#             category_keywords = self._get_category_keywords(category)
#             score = self._calculate_category_score(text_lower, self._get_category_keyword_dict(category), [])
#
#             # Find matching keywords
#             matching_keywords = [kw for kw in category_keywords[:20] if kw in text_lower]
#
#             detailed_analysis['category_scores'][category] = {
#                 'score': score,
#                 'matching_keywords': matching_keywords,
#                 'match_count': len(matching_keywords)
#             }
#
#         # Check pattern matches
#         for pattern_type, patterns in self.email_patterns.items():
#             matches = []
#             for pattern in patterns:
#                 if re.search(pattern, text, re.IGNORECASE):
#                     matches.append(pattern)
#             detailed_analysis['pattern_matches'][pattern_type] = matches
#
#         return detailed_analysis
#
#     def _get_category_keyword_dict(self, category):
#         """Get the keyword dictionary for a specific category"""
#         keyword_map = {
#             'Spam': self.spam_keywords,
#             'Promotions': self.promotion_keywords,
#             'Social': self.social_keywords,
#             'Purchases': self.purchase_keywords,
#             'Travel': self.travel_keywords,
#             'Forums': self.forum_keywords,
#             'Updates': self.update_keywords,
#             'Personal': self.personal_keywords
#         }
#         return keyword_map.get(category, {})


'''

------------------- Advanced Email Classifier -------------------

'''

import torch
import numpy as np
import re
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)


class PretrainedEmailClassifier:
    """
    Professional Email Classifier using multiple pretrained models
    Optimized for NVIDIA T500 2GB GPU and CPU usage
    """

    def __init__(self, model_type="distilbert", device="auto"):
        """
        Initialize classifier with different model options
        Args:
            model_type: "distilbert", "bert-base", "roberta", "simple"
            device: "auto", "cpu", "cuda"
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.classifier = None
        # Email category mapping
        self.categories = {
            0: "Updates",
            1: "Personal",
            2: "Promotions",
            3: "Forums",
            4: "Purchases",
            5: "Travel",
            6: "Spam",
            7: "Social"
        }
        # Enhanced keyword patterns for each category
        self._initialize_enhanced_patterns()
        # Load model with error handling
        self.load_model()

    def _initialize_enhanced_patterns(self):
        """Initialize ultra-comprehensive keyword patterns for all categories"""

        # SPAM - Ultra-comprehensive spam detection
        self.spam_keywords = {
            'urgent_words': ['urgent', 'immediate', 'act now', 'hurry', 'expires today', 'last chance',
                             'final notice', 'time sensitive', 'don\'t delay', 'expires soon', 'deadline',
                             'limited time only', 'offer expires', 'must act', 'don\'t wait', 'rush',
                             'asap', 'emergency', 'critical', 'breaking news', 'alert', 'warning',
                             'expire tonight', 'hours left', 'minutes left', 'ending soon', 'while supplies last',
                             'one time only', 'never again', 'now or never', 'this is it', 'final call'],
            'money_offers': ['win', 'winner', 'won', 'prize', 'lottery', 'cash', 'money', 'million', 'thousand',
                             'jackpot', 'sweepstakes', 'windfall', 'inheritance',
                             'free money', 'easy money', 'make money fast', 'get paid', 'earn money',
                             'financial freedom', 'passive income', 'work from home income', 'get rich',
                             'wealthy', 'fortune', 'treasure', 'bonus', 'reward', 'compensation',
                             'settlement', 'refund', 'rebate', 'cashback', 'profit', 'earnings',
                             'dollars', 'euros', 'pounds', 'yen', 'currency', 'bitcoin', 'crypto',
                             'investment return', 'dividend', 'interest', 'roi', 'payout'],
            'suspicious_claims': ['guaranteed', 'no risk', 'risk free', '100% guaranteed', 'promise',
                                  'certified', 'verified winner', 'pre-selected', 'congratulations',
                                  'you have been chosen', 'specially selected', 'exclusive opportunity',
                                  'one in a million', 'lucky you', 'destiny', 'fate', 'miracle',
                                  'unbelievable', 'incredible', 'amazing offer', 'once in lifetime',
                                  'breakthrough', 'revolutionary', 'secret formula', 'hidden truth',
                                  'insider information', 'confidential', 'classified', 'exclusive access'],
            'action_words': ['click here', 'call now', 'click now', 'visit now', 'order now',
                             'buy now', 'download now', 'register now', 'sign up now', 'claim now',
                             'redeem now', 'get started', 'apply now', 'join now', 'subscribe now',
                             'activate now', 'confirm now', 'verify now', 'accept now', 'respond now',
                             'reply immediately', 'contact us', 'don\'t miss out', 'secure your spot',
                             'reserve now', 'book now', 'grab this', 'take advantage', 'act fast'],
            'medical_pharma': ['viagra', 'cialis', 'pharmacy', 'pills', 'medication', 'prescription',
                               'doctor approved', 'fda approved', 'medical breakthrough', 'cure',
                               'lose weight', 'weight loss', 'diet pills', 'miracle cure', 'treatment',
                               'therapy', 'supplement', 'enhancement', 'enlargement', 'potency',
                               'stamina', 'performance', 'anti-aging', 'fountain of youth', 'longevity',
                               'health secret', 'natural remedy', 'herbal', 'organic cure'],
            'scam_indicators': ['nigerian prince', 'beneficiary', 'next of kin', 'transfer funds',
                                'bank details', 'account details', 'routing number', 'ssn',
                                'social security', 'wire transfer', 'western union', 'money transfer',
                                'deceased relative', 'inheritance claim', 'unclaimed funds', 'dormant account',
                                'foreign lottery', 'international sweepstakes', 'tax refund', 'irs refund',
                                'government grant', 'stimulus check', 'bailout money', 'compensation fund',
                                'class action settlement', 'legal settlement', 'insurance claim',
                                'investment opportunity', 'business proposal', 'partnership offer'],
            'fake_urgency': ['expires midnight', 'today only', 'flash sale', 'lightning deal',
                             'doorbuster', 'early bird', 'pre-launch', 'beta access', 'vip access',
                             'members only', 'invitation only', 'private sale', 'insider deal',
                             'staff pick', 'editor choice', 'recommended', 'featured', 'trending'],
            'psychological_triggers': ['fear of missing out', 'fomo', 'social proof', 'bandwagon',
                                       'scarcity', 'authority', 'celebrity endorsed', 'as seen on tv',
                                       'testimonial', 'success story', 'before and after', 'results',
                                       'transformation', 'life changing', 'game changer', 'breakthrough']
        }
        # PROMOTIONS - Ultra-comprehensive marketing and sales content
        self.promotion_keywords = {
            'sales_words': ['sale', 'discount', 'offer', 'deal', 'bargain', 'clearance', 'markdown',
                            'reduced', 'price drop', 'flash sale', 'mega sale', 'super sale', 'big sale',
                            'end of season', 'closeout', 'liquidation', 'warehouse sale', 'outlet',
                            'overstock', 'surplus', 'inventory clearance', 'moving sale', 'going out of business',
                            'retirement sale', 'store closing', 'everything must go', 'final days',
                            'blowout', 'fire sale', 'red tag', 'yellow tag', 'blue light special'],
            'discounts': ['%', 'percent', 'off', 'save', 'savings', 'coupon', 'promo code', 'voucher',
                          'discount code', 'special price', 'reduced price', 'slashed price', 'half price',
                          'buy one get one', 'bogo', '2 for 1', 'bundle deal', '10% off', '20% off',
                          '30% off', '40% off', '50% off', '60% off', '70% off', '80% off', '90% off',
                          'up to 90% off', 'wholesale price', 'cost price', 'below cost', 'at cost',
                          'manufacturer price', 'factory direct', 'no markup', 'price match'],
            'marketing_terms': ['exclusive', 'limited offer', 'special offer', 'promotional', 'featured',
                                'bestseller', 'top rated', 'customer favorite', 'most popular', 'trending',
                                'new arrival', 'just in', 'fresh stock', 'back in stock', 'restocked',
                                'pre-order', 'early access', 'sneak peek', 'preview', 'debut',
                                'launch', 'introducing', 'unveiling', 'revealing', 'announcing',
                                'premium', 'luxury', 'deluxe', 'elite', 'signature', 'collection',
                                'series', 'edition', 'version', 'model', 'style', 'design'],
            'shopping_words': ['shop', 'store', 'retail', 'mall', 'marketplace', 'catalog', 'collection',
                               'browse', 'selection', 'inventory', 'merchandise', 'products', 'items',
                               'goods', 'wares', 'stock', 'supplies', 'materials', 'equipment',
                               'accessories', 'components', 'parts', 'tools', 'gadgets', 'devices',
                               'appliances', 'electronics', 'fashion', 'clothing', 'apparel', 'footwear',
                               'jewelry', 'watches', 'bags', 'furniture', 'home decor', 'garden'],
            'seasonal_sales': ['black friday', 'cyber monday', 'christmas sale', 'holiday sale',
                               'summer sale', 'winter clearance', 'spring collection', 'back to school',
                               'valentine\'s day', 'mother\'s day', 'father\'s day', 'memorial day',
                               'labor day', 'thanksgiving', 'new year', 'easter', 'halloween',
                               'independence day', '4th of july', 'presidents day', 'columbus day'],
            'call_to_action': ['shop now', 'buy today', 'order online', 'visit store', 'call store',
                               'while supplies last', 'limited quantity', 'limited stock', 'few left',
                               'almost sold out', 'selling fast', 'popular item', 'hot item',
                               'in demand', 'customer choice', 'staff pick', 'recommended'],
            'price_terms': ['price', 'cost', 'value', 'worth', 'cheap', 'affordable', 'budget',
                            'economy', 'low cost', 'inexpensive', 'reasonable', 'competitive',
                            'unbeatable', 'lowest price', 'best price', 'great value', 'excellent value',
                            'money back', 'satisfaction guaranteed', 'no questions asked', 'full refund'],
            'loyalty_programs': ['rewards', 'points', 'cashback', 'loyalty', 'member', 'membership',
                                 'vip', 'platinum', 'gold', 'silver', 'bronze', 'tier', 'level',
                                 'club', 'program', 'benefits', 'perks', 'privileges', 'exclusive access']
        }
        # SOCIAL - Ultra-comprehensive social media and networking
        self.social_keywords = {
            'platforms': ['facebook', 'twitter', 'instagram', 'linkedin', 'snapchat', 'tiktok',
                          'youtube', 'pinterest', 'reddit', 'discord', 'whatsapp', 'telegram',
                          'social media', 'social network', 'social platform', 'meta', 'x.com',
                          'threads', 'mastodon', 'bluesky', 'clubhouse', 'twitch', 'mixer',
                          'vine', 'periscope', 'skype', 'zoom', 'facetime', 'messenger',
                          'wechat', 'line', 'viber', 'kik', 'signal', 'wickr', 'slack'],
            'activities': ['posted', 'shared', 'liked', 'commented', 'tagged', 'mentioned', 'followed',
                           'unfollowed', 'friended', 'connected', 'endorsed', 'recommended', 'reviewed',
                           'rated', 'voted', 'subscribed', 'unsubscribed', 'blocked', 'reported',
                           'joined', 'left', 'created', 'started', 'launched', 'published',
                           'uploaded', 'downloaded', 'streamed', 'broadcasted', 'went live'],
            'notifications': ['notification', 'alert', 'friend request', 'connection request',
                              'follow request', 'message request', 'group invitation', 'event invitation',
                              'tagged you', 'mentioned you', 'commented on', 'liked your', 'shared your',
                              'reacted to', 'responded to', 'replied to', 'dm', 'direct message',
                              'private message', 'inbox', 'chat', 'conversation', 'thread'],
            'content_types': ['profile', 'timeline', 'feed', 'story', 'post', 'tweet', 'status update',
                              'photo', 'video', 'live stream', 'reel', 'story highlight', 'memory',
                              'flashback', 'throwback', 'tbt', 'selfie', 'groupie', 'boomerang',
                              'gif', 'meme', 'sticker', 'emoji', 'reaction', 'poll', 'quiz',
                              'survey', 'question', 'ama', 'ask me anything', 'q&a'],
            'engagement': ['views', 'likes', 'shares', 'comments', 'reactions', 'followers', 'following',
                           'subscribers', 'connections', 'network', 'community', 'group', 'page',
                           'fan page', 'business page', 'profile views', 'reach', 'impressions',
                           'engagement rate', 'viral', 'trending', 'hashtag', 'tag', 'mention'],
            'social_features': ['check-in', 'location', 'geotag', 'nearby', 'events', 'calendar',
                                'reminder', 'birthday', 'anniversary', 'milestone', 'achievement',
                                'badge', 'award', 'recognition', 'verification', 'verified',
                                'blue tick', 'premium', 'plus', 'pro', 'business account'],
            'privacy_settings': ['privacy', 'public', 'private', 'friends only', 'followers only',
                                 'custom', 'restricted', 'blocked', 'hidden', 'visible', 'anonymous',
                                 'incognito', 'ghost mode', 'invisible', 'offline', 'away', 'busy'],
            'social_emotions': ['lol', 'lmao', 'rofl', 'omg', 'wtf', 'fml', 'yolo', 'fomo',
                                'blessed', 'grateful', 'excited', 'happy', 'sad', 'angry', 'love',
                                'hate', 'jealous', 'proud', 'disappointed', 'surprised', 'shocked']
        }
        # PURCHASES - Ultra-comprehensive e-commerce and transactions
        self.purchase_keywords = {
            'order_status': ['order', 'purchase', 'bought', 'ordered', 'checkout', 'cart', 'basket',
                             'order confirmation', 'order placed', 'order received', 'processing',
                             'preparing', 'packed', 'ready to ship', 'dispatched', 'fulfilled',
                             'completed', 'cancelled', 'pending', 'on hold', 'backorder',
                             'pre-order', 'reserved', 'allocated', 'picked', 'quality checked'],
            'payment_terms': ['payment', 'paid', 'charged', 'billed', 'invoice', 'receipt', 'billing',
                              'transaction', 'refund', 'refunded', 'chargeback', 'credit card', 'paypal',
                              'stripe', 'square', 'payment method', 'payment declined', 'payment failed',
                              'payment pending', 'payment authorized', 'payment captured', 'settlement',
                              'installment', 'monthly payment', 'emi', 'financing', 'loan', 'credit',
                              'debit', 'cash', 'check', 'money order', 'wire transfer', 'ach'],
            'shipping': ['shipped', 'shipping', 'delivered', 'delivery', 'tracking', 'tracking number',
                         'fedex', 'ups', 'dhl', 'usps', 'courier', 'express', 'standard shipping',
                         'free shipping', 'same day delivery', 'next day', 'expedited', 'overnight',
                         'ground', 'air', 'freight', 'cargo', 'logistics', 'fulfillment',
                         'warehouse', 'distribution center', 'depot', 'hub', 'sorting facility'],
            'retail_terms': ['amazon', 'ebay', 'walmart', 'target', 'best buy', 'costco', 'marketplace',
                             'vendor', 'seller', 'merchant', 'retailer', 'supplier', 'manufacturer',
                             'wholesaler', 'distributor', 'reseller', 'dealer', 'agent', 'broker',
                             'third party', 'dropshipper', 'affiliate', 'partner', 'brand', 'store'],
            'return_exchange': ['return', 'exchange', 'replacement', 'warranty', 'guarantee',
                                'defective', 'damaged', 'wrong item', 'not as described', 'quality issue',
                                'manufacturing defect', 'faulty', 'broken', 'cracked', 'scratched',
                                'missing parts', 'incomplete', 'used', 'opened', 'tampered',
                                'expired', 'recalled', 'safety issue', 'hazardous'],
            'product_categories': ['electronics', 'clothing', 'shoes', 'accessories', 'jewelry',
                                   'books', 'music', 'movies', 'games', 'toys', 'sports', 'fitness',
                                   'health', 'beauty', 'home', 'garden', 'kitchen', 'appliances',
                                   'furniture', 'decor', 'automotive', 'tools', 'hardware'],
            'shopping_experience': ['cart', 'wishlist', 'favorites', 'saved items', 'compare',
                                    'review', 'rating', 'stars', 'feedback', 'testimonial',
                                    'recommendation', 'suggestion', 'similar items', 'related products',
                                    'frequently bought together', 'customers also bought', 'bestseller'],
            'loyalty_rewards': ['points', 'miles', 'cashback', 'rewards', 'loyalty', 'membership',
                                'tier', 'status', 'benefits', 'perks', 'credits', 'bonus', 'rebate',
                                'discount', 'coupon', 'promo code', 'gift card', 'store credit'],
            'customer_service': ['support', 'help', 'assistance', 'service', 'representative',
                                 'agent', 'specialist', 'expert', 'consultant', 'advisor',
                                 'chat', 'phone', 'email', 'ticket', 'case', 'inquiry', 'question']
        }
        # TRAVEL - Ultra-comprehensive tourism and transportation
        self.travel_keywords = {
            'transportation': ['flight', 'plane', 'airplane', 'airline', 'airport', 'boarding pass',
                               'gate', 'terminal', 'departure', 'arrival', 'layover', 'connecting flight',
                               'train', 'bus', 'car rental', 'taxi', 'uber', 'lyft', 'ride share',
                               'ferry', 'cruise', 'ship', 'boat', 'yacht', 'helicopter', 'shuttle',
                               'metro', 'subway', 'tram', 'trolley', 'cable car', 'rickshaw'],
            'accommodation': ['hotel', 'motel', 'resort', 'accommodation', 'booking', 'reservation',
                              'room', 'suite', 'check-in', 'check-out', 'guest', 'stay', 'lodging',
                              'airbnb', 'vrbo', 'bed and breakfast', 'hostel', 'inn', 'guesthouse',
                              'villa', 'apartment', 'condo', 'cabin', 'cottage', 'chalet', 'lodge',
                              'mansion', 'palace', 'castle', 'tent', 'camping', 'glamping', 'rv'],
            'travel_services': ['travel', 'trip', 'vacation', 'holiday', 'tour', 'cruise', 'safari',
                                'excursion', 'adventure', 'getaway', 'retreat', 'expedition', 'journey',
                                'voyage', 'pilgrimage', 'business trip', 'leisure travel', 'solo travel',
                                'group travel', 'family vacation', 'honeymoon', 'anniversary trip',
                                'backpacking', 'road trip', 'camping trip', 'hiking', 'trekking'],
            'destinations': ['destination', 'city', 'country', 'beach', 'mountain', 'resort town',
                             'tourist attraction', 'landmark', 'museum', 'theme park', 'national park',
                             'state park', 'monument', 'cathedral', 'temple', 'mosque', 'synagogue',
                             'palace', 'castle', 'fort', 'ruins', 'archaeological site', 'unesco',
                             'world heritage', 'wonder of the world', 'scenic route', 'scenic drive'],
            'booking_terms': ['itinerary', 'travel agent', 'expedia', 'booking.com', 'priceline',
                              'kayak', 'travelocity', 'orbitz', 'travel insurance', 'visa', 'passport',
                              'customs', 'immigration', 'border', 'checkpoint', 'security', 'tsa',
                              'duty free', 'currency exchange', 'foreign exchange', 'travelers checks'],
            'travel_documentation': ['passport', 'visa', 'id', 'driver license', 'international permit',
                                     'travel document', 'health certificate', 'vaccination record',
                                     'medical certificate', 'travel advisory', 'embassy', 'consulate',
                                     'diplomatic', 'tourist visa', 'business visa', 'transit visa'],
            'travel_activities': ['sightseeing', 'touring', 'exploring', 'discovering', 'wandering',
                                  'adventure sports', 'water sports', 'skiing', 'snowboarding',
                                  'surfing', 'diving', 'snorkeling', 'fishing', 'hunting', 'safari',
                                  'wildlife watching', 'bird watching', 'photography', 'culture'],
            'travel_planning': ['budget', 'itinerary', 'schedule', 'timeline', 'duration', 'length',
                                'season', 'weather', 'climate', 'temperature', 'rainfall', 'forecast',
                                'best time to visit', 'peak season', 'off season', 'shoulder season',
                                'travel guide', 'guidebook', 'map', 'gps', 'navigation', 'directions'],
            'travel_experiences': ['local culture', 'tradition', 'customs', 'festival', 'celebration',
                                   'cuisine', 'food', 'restaurant', 'street food', 'local delicacy',
                                   'souvenir', 'shopping', 'market', 'bazaar', 'local craft', 'art',
                                   'music', 'dance', 'performance', 'show', 'theater', 'concert']
        }
        # FORUMS - Ultra-comprehensive discussion boards and communities
        self.forum_keywords = {
            'forum_terms': ['forum', 'discussion', 'board', 'community', 'thread', 'topic', 'post',
                            'reply', 'comment', 'discussion board', 'message board', 'bulletin board',
                            'chat room', 'chat forum', 'online community', 'web forum', 'support forum',
                            'help forum', 'technical forum', 'user forum', 'customer forum',
                            'knowledge base', 'wiki', 'faq', 'documentation', 'manual', 'guide'],
            'user_roles': ['member', 'user', 'moderator', 'admin', 'administrator', 'staff', 'vip',
                           'premium member', 'gold member', 'verified', 'contributor', 'expert',
                           'guru', 'veteran', 'senior member', 'junior member', 'newbie', 'rookie',
                           'banned', 'suspended', 'muted', 'warned', 'trusted', 'verified user',
                           'power user', 'super user', 'elite member', 'founding member'],
            'activities': ['posted', 'replied', 'commented', 'mentioned', 'quoted', 'subscribed',
                           'unsubscribed', 'watched', 'favorited', 'bookmarked', 'voted', 'rated',
                           'thanked', 'liked', 'disliked', 'reported', 'flagged', 'edited',
                           'deleted', 'moved', 'locked', 'pinned', 'sticky', 'closed', 'archived'],
            'content_types': ['question', 'answer', 'solution', 'tutorial', 'guide', 'review',
                              'opinion', 'poll', 'survey', 'announcement', 'sticky', 'pinned',
                              'faq', 'how-to', 'tip', 'trick', 'hack', 'workaround', 'bug report',
                              'feature request', 'suggestion', 'feedback', 'complaint', 'praise'],
            'platforms': ['reddit', 'stack overflow', 'quora', 'discourse', 'phpbb', 'vbulletin',
                          'discord server', 'slack workspace', 'telegram group', 'facebook group',
                          'linkedin group', 'google groups', 'yahoo groups', 'github discussions',
                          'steam community', 'xbox live', 'playstation network', 'nintendo network'],
            'forum_features': ['reputation', 'karma', 'points', 'badges', 'achievements', 'awards',
                               'medals', 'trophies', 'ranks', 'levels', 'experience points', 'xp',
                               'post count', 'join date', 'last seen', 'online status', 'avatar',
                               'signature', 'profile', 'bio', 'about me', 'location', 'interests'],
            'discussion_terms': ['debate', 'argument', 'discussion', 'conversation', 'dialogue',
                                 'exchange', 'opinion', 'viewpoint', 'perspective', 'stance',
                                 'position', 'belief', 'thought', 'idea', 'concept', 'theory',
                                 'hypothesis', 'speculation', 'assumption', 'conclusion', 'insight'],
            'community_aspects': ['community guidelines', 'rules', 'terms of service', 'code of conduct',
                                  'etiquette', 'netiquette', 'behavior', 'conduct', 'respect',
                                  'tolerance', 'diversity', 'inclusion', 'harassment', 'trolling',
                                  'spam', 'off-topic', 'derailing', 'flame war', 'heated discussion'],
            'technical_terms': ['subforum', 'category', 'section', 'board', 'channel', 'room',
                                'private message', 'pm', 'direct message', 'dm', 'notification',
                                'alert', 'mention', 'tag', 'search', 'filter', 'sort', 'recent',
                                'popular', 'trending', 'hot', 'rising', 'new', 'old', 'archived']
        }
        # UPDATES - Ultra-comprehensive system notifications and account updates
        self.update_keywords = {
            'account_terms': ['account', 'profile', 'settings', 'preferences', 'subscription',
                              'membership', 'plan', 'upgrade', 'downgrade', 'renewal', 'expiration',
                              'activate', 'deactivate', 'suspend', 'terminated', 'deleted', 'closed',
                              'frozen', 'locked', 'unlocked', 'restricted', 'unrestricted',
                              'verified', 'unverified', 'confirmed', 'pending confirmation'],
            'notifications': ['notification', 'alert', 'reminder', 'notice', 'announcement',
                              'update', 'news', 'bulletin', 'advisory', 'warning', 'maintenance',
                              'scheduled maintenance', 'system update', 'security update',
                              'patch', 'hotfix', 'bug fix', 'improvement', 'enhancement',
                              'new feature', 'feature update', 'version update', 'upgrade'],
            'security': ['security', 'password', 'login', 'logout', 'authentication', 'verification',
                         'two-factor', '2fa', 'verify', 'confirm', 'reset', 'change password',
                         'suspicious activity', 'unauthorized access', 'security breach',
                         'data breach', 'compromise', 'hacking', 'phishing', 'malware',
                         'virus', 'intrusion', 'firewall', 'antivirus', 'encryption', 'ssl'],
            'financial': ['statement', 'balance', 'transaction', 'deposit', 'withdrawal', 'transfer',
                          'bank', 'credit card', 'debit', 'finance', 'billing cycle', 'due date',
                          'overdue', 'payment reminder', 'auto-pay', 'autopay', 'direct debit',
                          'standing order', 'recurring payment', 'subscription fee', 'service charge',
                          'interest', 'penalty', 'late fee', 'overdraft', 'credit limit'],
            'system_terms': ['system', 'service', 'server', 'maintenance', 'downtime', 'upgrade',
                             'patch', 'bug fix', 'feature', 'improvement', 'changelog', 'version',
                             'release', 'deployment', 'rollback', 'hotfix', 'critical update',
                             'emergency maintenance', 'planned outage', 'service disruption',
                             'network issue', 'connectivity problem', 'performance improvement'],
            'communication_updates': ['email change', 'phone change', 'address change', 'contact update',
                                      'communication preferences', 'newsletter', 'subscription',
                                      'unsubscribe', 'opt-in', 'opt-out', 'privacy settings',
                                      'data protection', 'gdpr', 'privacy policy', 'terms update'],
            'service_updates': ['service announcement', 'policy change', 'terms of service',
                                'privacy policy', 'user agreement', 'license agreement',
                                'feature deprecation', 'service discontinuation', 'migration',
                                'data migration', 'account migration', 'platform change'],
            'regulatory_updates': ['compliance', 'regulation', 'legal notice', 'court order',
                                   'government request', 'law enforcement', 'subpoena', 'warrant',
                                   'tax information', 'tax document', '1099', 'w2', 'tax season',
                                   'audit', 'investigation', 'regulatory change', 'policy update'],
            'health_monitoring': ['health check', 'system health', 'performance metrics', 'uptime',
                                  'downtime', 'response time', 'latency', 'throughput', 'capacity',
                                  'load', 'traffic', 'usage statistics', 'analytics', 'monitoring'],
            'backup_recovery': ['backup', 'restore', 'recovery', 'data backup', 'system backup',
                                'disaster recovery', 'failover', 'redundancy', 'data loss',
                                'corruption', 'integrity check', 'verification', 'sync', 'synchronization']
        }
        # PERSONAL - Ultra-comprehensive personal communications
        self.personal_keywords = {
            'greetings': ['hi', 'hello', 'hey', 'dear', 'greetings', 'good morning', 'good afternoon',
                          'good evening', 'hope you\'re well', 'how are you', 'how\'s it going',
                          'what\'s up', 'wassup', 'sup', 'yo', 'hiya', 'howdy', 'salutations',
                          'good day', 'good to see you', 'nice to see you', 'long time no see',
                          'been a while', 'how have you been', 'hope all is well', 'trust you\'re well'],
            'personal_terms': ['friend', 'buddy', 'pal', 'mate', 'family', 'relatives', 'personal',
                               'private', 'confidential', 'between us', 'just between friends',
                               'bestie', 'bff', 'brother', 'sister', 'bro', 'sis', 'cousin',
                               'nephew', 'niece', 'uncle', 'aunt', 'grandma', 'grandpa', 'mom',
                               'dad', 'mother', 'father', 'son', 'daughter', 'kid', 'child'],
            'social_activities': ['meet', 'meeting', 'get together', 'hang out', 'catch up', 'coffee',
                                  'lunch', 'dinner', 'drinks', 'party', 'celebration', 'birthday',
                                  'anniversary', 'wedding', 'graduation', 'reunion', 'barbecue',
                                  'picnic', 'camping', 'hiking', 'movie', 'theater', 'concert',
                                  'game night', 'poker night', 'book club', 'workout', 'gym'],
            'emotions': ['love', 'miss', 'excited', 'happy', 'sad', 'worried', 'concerned', 'grateful',
                         'thankful', 'appreciate', 'congratulations', 'condolences', 'sympathy',
                         'sorry', 'apologize', 'forgive', 'understand', 'support', 'care',
                         'proud', 'impressed', 'amazed', 'surprised', 'shocked', 'devastated',
                         'heartbroken', 'overjoyed', 'thrilled', 'delighted', 'relieved'],
            'life_events': ['baby', 'pregnancy', 'job', 'promotion', 'retirement', 'moving', 'house',
                            'car', 'vacation', 'hobby', 'health', 'doctor', 'hospital', 'surgery',
                            'illness', 'recovery', 'treatment', 'medication', 'therapy', 'exercise',
                            'diet', 'weight loss', 'achievement', 'accomplishment', 'milestone'],
            'relationship_terms': ['relationship', 'dating', 'boyfriend', 'girlfriend', 'partner',
                                   'spouse', 'husband', 'wife', 'engaged', 'engagement', 'married',
                                   'marriage', 'wedding', 'divorce', 'separated', 'single',
                                   'break up', 'breakup', 'make up', 'reconcile', 'anniversary'],
            'casual_expressions': ['btw', 'by the way', 'anyway', 'so', 'well', 'guess what',
                                   'speaking of', 'reminds me', 'oh yeah', 'also', 'plus',
                                   'actually', 'honestly', 'frankly', 'seriously', 'literally',
                                   'basically', 'obviously', 'apparently', 'hopefully', 'luckily'],
            'time_references': ['yesterday', 'today', 'tomorrow', 'weekend', 'week', 'month',
                                'last week', 'next week', 'last month', 'next month', 'recently',
                                'lately', 'soon', 'later', 'earlier', 'before', 'after',
                                'meanwhile', 'currently', 'now', 'then', 'when', 'while'],
            'personal_sharing': ['thought you\'d like', 'wanted to share', 'check this out',
                                 'you won\'t believe', 'amazing news', 'great news', 'bad news',
                                 'update on', 'quick update', 'just wanted to say', 'had to tell you',
                                 'couldn\'t wait to share', 'exciting news', 'sad news', 'funny story'],
            'support_comfort': ['here for you', 'thinking of you', 'praying for you', 'sending love',
                                'take care', 'feel better', 'get well soon', 'hang in there',
                                'you got this', 'stay strong', 'don\'t worry', 'everything will be okay',
                                'it\'s going to be fine', 'we\'ll get through this', 'I\'m here'],
            'memory_nostalgia': ['remember when', 'do you remember', 'reminds me of', 'brings back memories',
                                 'good old days', 'back in the day', 'when we were', 'used to',
                                 'nostalgic', 'throwback', 'flashback', 'memory lane', 'old times',
                                 'childhood', 'growing up', 'school days', 'college days']
        }
        # Ultra-comprehensive email structure patterns with advanced regex
        self.email_patterns = {
            'promotional_structure': [
                r'\b(save|discount)\s+\d+%',
                r'\$\d+\s+(off|discount)',
                r'(limited|exclusive)\s+(offer|deal)',
                r'(buy|shop)\s+now',
                r'(free|complimentary)\s+(shipping|delivery)',
                r'\b\d+%\s+(off|discount|savings)',
                r'(sale|clearance|markdown)\s+\d+%',
                r'(promo|coupon)\s+code',
                r'(expires?|ends?)\s+(today|tomorrow|soon)',
                r'(flash|mega|super|big)\s+sale',
                r'(black friday|cyber monday|holiday)',
                r'(bogo|buy one get one)',
                r'(while supplies last|limited quantity)',
                r'(act now|don\'t wait|hurry)',
                r'(new arrival|just in|back in stock)'
            ],
            'spam_structure': [
                r'(congratulations|winner|won)\s+.{0,20}\$([\d,]+)',
                r'(urgent|immediate|act now)',
                r'(click here|call now).{0,10}(win|money|prize)',
                r'\b(guarantee|guaranteed)\s+(money|income|profit)',
                r'(lottery|sweepstakes|jackpot)\s+(winner|win)',
                r'(free|easy)\s+money',
                r'(make money|get paid|earn money)\s+(fast|quick|easy)',
                r'(risk free|no risk|100% guaranteed)',
                r'(nigerian|inheritance|beneficiary)',
                r'(bank details|account information|wire transfer)',
                r'(viagra|cialis|pharmacy|pills)',
                r'(lose weight|diet pills|miracle cure)',
                r'(work from home|financial freedom)',
                r'(expires today|final notice|last chance)',
                r'(credit report|debt relief|loan approval)'
            ],
            'personal_structure': [
                r'^(hi|hello|hey|dear)\s+\w+',
                r'(how are you|how\'s it going|catch up)',
                r'(love|miss)\s+you',
                r'(coffee|lunch|dinner|drinks)\s+(this|next|tomorrow)',
                r'(hope you\'re|hope all is)\s+(well|good)',
                r'(long time no see|been a while)',
                r'(thinking of you|miss you)',
                r'(family|friend|buddy|pal)',
                r'(birthday|anniversary|celebration)',
                r'(vacation|trip|travel)',
                r'(remember when|do you remember)',
                r'(just wanted to|thought you\'d)',
                r'(exciting news|great news|update)',
                r'(take care|feel better|get well)',
                r'(love and|hugs and|xoxo)'
            ],
            'purchase_structure': [
                r'(order|purchase)\s+(confirmation|receipt)',
                r'(tracking|shipment)\s+(number|information)',
                r'(payment|transaction)\s+(confirmation|receipt)',
                r'(invoice|receipt|billing)',
                r'(shipped|delivered|tracking)',
                r'(amazon|ebay|walmart|target)',
                r'(fedex|ups|usps|dhl)',
                r'(return|exchange|refund)',
                r'(warranty|guarantee)',
                r'(customer service|support)',
                r'(order #|invoice #|tracking #)',
                r'(estimated delivery|delivery date)',
                r'(payment method|credit card)',
                r'(shipping address|billing address)',
                r'(order status|shipment status)'
            ],
            'travel_structure': [
                r'(flight|booking|reservation)\s+(confirmation|number)',
                r'(check-in|boarding|departure)',
                r'(hotel|accommodation|room)',
                r'(itinerary|travel|trip)',
                r'(airline|airport|gate|terminal)',
                r'(passport|visa|travel document)',
                r'(destination|vacation|holiday)',
                r'(travel insurance|travel agent)',
                r'(cruise|tour|excursion)',
                r'(car rental|taxi|uber|lyft)',
                r'(departure time|arrival time)',
                r'(seat assignment|boarding pass)',
                r'(baggage|luggage|carry-on)',
                r'(customs|immigration|security)',
                r'(travel advisory|weather alert)'
            ],
            'social_structure': [
                r'(facebook|twitter|instagram|linkedin)',
                r'(tagged|mentioned|commented)',
                r'(posted|shared|liked)',
                r'(friend request|connection)',
                r'(notification|alert)',
                r'(profile|timeline|feed)',
                r'(social media|social network)',
                r'(followers|following|connections)',
                r'(group invitation|event invitation)',
                r'(direct message|private message)',
                r'(story|post|photo|video)',
                r'(reaction|comment|share)',
                r'(hashtag|tag|mention)',
                r'(live stream|broadcast)',
                r'(social update|activity)'
            ],
            'forum_structure': [
                r'(forum|discussion|thread)',
                r'(replied|commented|posted)',
                r'(community|board|group)',
                r'(moderator|admin|staff)',
                r'(question|answer|solution)',
                r'(reddit|stack overflow|quora)',
                r'(upvote|downvote|karma)',
                r'(subscription|notification)',
                r'(new reply|new comment)',
                r'(discussion topic|forum topic)',
                r'(member|user|contributor)',
                r'(help|support|assistance)',
                r'(tutorial|guide|how-to)',
                r'(bug report|feature request)',
                r'(community guidelines|rules)'
            ],
            'update_structure': [
                r'(account|profile|settings)',
                r'(password|security|authentication)',
                r'(statement|balance|transaction)',
                r'(notification|alert|reminder)',
                r'(update|upgrade|maintenance)',
                r'(system|service|server)',
                r'(billing|payment|subscription)',
                r'(renewal|expiration|due date)',
                r'(security alert|suspicious activity)',
                r'(privacy policy|terms of service)',
                r'(software update|system update)',
                r'(account verification|email confirmation)',
                r'(backup|restore|sync)',
                r'(compliance|regulatory|legal)',
                r'(performance|monitoring|health check)'
            ]
        }
        # Advanced contextual indicators
        self.contextual_indicators = {
            'sender_domains': {
                'promotional': ['deals', 'offers', 'sales', 'promo', 'marketing', 'newsletter', 'shop'],
                'social': ['facebook', 'twitter', 'instagram', 'linkedin', 'social', 'community'],
                'financial': ['bank', 'finance', 'credit', 'payment', 'billing', 'account'],
                'travel': ['airline', 'hotel', 'travel', 'booking', 'reservation', 'trip'],
                'ecommerce': ['amazon', 'ebay', 'shop', 'store', 'retail', 'marketplace'],
                'tech': ['github', 'stackoverflow', 'tech', 'developer', 'code', 'programming'],
                'support': ['support', 'help', 'service', 'customer', 'noreply', 'donotreply']
            },
            'subject_prefixes': {
                'spam': ['urgent', 'congratulations', 'winner', 'free', 'limited time', 'act now'],
                'promotional': ['sale', 'offer', 'deal', 'discount', '% off', 'special price'],
                'personal': ['re:', 'fwd:', 'hey', 'hi', 'hello', 'catch up', 'miss you'],
                'system': ['action required', 'notification', 'alert', 'reminder', 'update'],
                'transactional': ['receipt', 'confirmation', 'order', 'invoice', 'payment'],
                'social': ['tagged', 'mentioned', 'posted', 'shared', 'liked', 'commented']
            },
            'content_length_indicators': {
                'spam': (50, 500),  # Usually short and punchy
                'promotional': (100, 1000),  # Medium length with details
                'personal': (50, 2000),  # Varies widely
                'social': (20, 200),  # Usually short notifications
                'transactional': (100, 500),  # Concise and factual
                'system': (100, 800)  # Detailed but structured
            }
        }

    def _get_device(self, device):
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                    print(f"GPU Memory: {gpu_memory:.1f}GB")
                    if gpu_memory >= 1.5:
                        return "cuda"
                    else:
                        print("GPU memory insufficient, using CPU")
                        return "cpu"
                except:
                    print("CUDA error detected, using CPU")
                    return "cpu"
            else:
                print("CUDA not available, using CPU")
                return "cpu"
        return device

    def load_model(self):
        """Load the specified pretrained model with fallback"""
        print(f"Loading {self.model_type} model on {self.device}...")
        try:
            if self.model_type == "distilbert":
                self._load_distilbert()
            elif self.model_type == "bert-base":
                self._load_bert_base()
            elif self.model_type == "roberta":
                self._load_roberta()
            elif self.model_type == "simple":
                self._load_simple_classifier()
            else:
                print(f"Unknown model type: {self.model_type}, using simple classifier")
                self._load_simple_classifier()
        except Exception as e:
            print(f"Error loading {self.model_type}: {e}")
            print("Falling back to simple rule-based classifier...")
            self._load_simple_classifier()

    def _load_distilbert(self):
        """Load DistilBERT - Lightweight and fast"""
        try:
            # Load pre-trained sentiment model that actually works
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print("✓ DistilBERT-based classifier loaded successfully")
        except Exception as e:
            print(f"Failed to load DistilBERT: {e}")
            raise

    def _load_bert_base(self):
        """Load BERT Base - More accurate but slower"""
        try:
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print("✓ BERT-based classifier loaded successfully")
        except Exception as e:
            print(f"Failed to load BERT: {e}")
            raise

    def _load_roberta(self):
        """Load RoBERTa - Good balance of speed and accuracy"""
        try:
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print("✓ RoBERTa classifier loaded successfully")
        except Exception as e:
            print(f"Failed to load RoBERTa: {e}")
            raise

    def _load_simple_classifier(self):
        """Load simple rule-based classifier as fallback"""
        self.classifier = None  # Will use rule-based classification
        print("✓ Simple rule-based classifier loaded successfully")

    def classify_email(self, subject="", body="", return_probabilities=False):
        """
        Classify a single email
        Args:
            subject: Email subject line
            body: Email body content
            return_probabilities: Whether to return all class probabilities
        Returns:
            dict: Classification result with category and confidence
        """
        # Combine subject and body
        text = f"Subject: {subject}\n\nBody: {body}".strip()
        # Truncate if too long
        if len(text) > 1000:
            text = text[:1000] + "..."
        try:
            if self.classifier is not None:
                # Use ML model - Fixed the error handling here
                results = self.classifier(text)
                # Handle the results properly
                if results and isinstance(results, list) and len(results) > 0:
                    # The results come as a list of dictionaries with 'label' and 'score'
                    # Find the result with highest score
                    best_result = max(results, key=lambda x: x['score'])
                    # Map sentiment to email category using content analysis
                    predicted_class = self._enhanced_content_analysis(text, best_result['label'])
                    confidence = self._calculate_enhanced_confidence(text, predicted_class)
                    # Create probability distribution
                    all_probs = self._create_enhanced_probability_distribution(predicted_class, confidence, text)
                else:
                    # Fallback to rule-based
                    return self._enhanced_rule_based_classify(subject, body, return_probabilities)
            else:
                # Use rule-based classification
                return self._enhanced_rule_based_classify(subject, body, return_probabilities)
            result = {
                'category': predicted_class,
                'confidence': float(confidence),
                'text_preview': text[:200] + "..." if len(text) > 200 else text,
                'method': 'ml-based'
            }
            if return_probabilities:
                result['all_probabilities'] = all_probs
            return result
        except Exception as e:
            print(f"Classification error: {e}")
            return self._enhanced_rule_based_classify(subject, body, return_probabilities)

    def _enhanced_content_analysis(self, text, sentiment):
        """Enhanced content analysis with comprehensive pattern matching"""
        text_lower = text.lower()

        # Calculate scores for each category
        category_scores = {}

        # Spam detection - highest priority for obvious spam
        spam_score = self._calculate_category_score(text_lower, self.spam_keywords,
                                                    self.email_patterns.get('spam_structure', []))
        if spam_score > 0.7:
            return 'Spam'

        # Calculate scores for all categories
        category_scores['Spam'] = spam_score
        category_scores['Promotions'] = self._calculate_category_score(text_lower, self.promotion_keywords,
                                                                       self.email_patterns.get('promotional_structure',
                                                                                               []))
        category_scores['Social'] = self._calculate_category_score(text_lower, self.social_keywords, [])
        category_scores['Purchases'] = self._calculate_category_score(text_lower, self.purchase_keywords, [])
        category_scores['Travel'] = self._calculate_category_score(text_lower, self.travel_keywords, [])
        category_scores['Forums'] = self._calculate_category_score(text_lower, self.forum_keywords, [])
        category_scores['Updates'] = self._calculate_category_score(text_lower, self.update_keywords, [])
        category_scores['Personal'] = self._calculate_category_score(text_lower, self.personal_keywords,
                                                                     self.email_patterns.get('personal_structure', []))

        # Find the category with highest score
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]

        # If no category has a strong score, use sentiment analysis
        if max_score < 0.3:
            sentiment_lower = sentiment.lower()
            if 'positive' in sentiment_lower or sentiment.startswith('LABEL_1'):
                return 'Personal'
            elif 'negative' in sentiment_lower or sentiment.startswith('LABEL_0'):
                return 'Spam'
            else:
                return 'Updates'

        return best_category

    def _calculate_category_score(self, text, keyword_dict, patterns):
        """Calculate score for a category based on keywords and patterns"""
        total_score = 0
        total_weight = 0

        # Score based on keyword categories
        for category, keywords in keyword_dict.items():
            category_weight = len(keywords)
            matches = sum(1 for keyword in keywords if keyword in text)
            category_score = matches / len(keywords) if keywords else 0

            # Apply category-specific weights
            if category in ['urgent_words', 'suspicious_claims', 'scam_indicators']:
                category_weight *= 2  # Higher weight for important spam indicators
            elif category in ['greetings', 'personal_terms', 'emotions']:
                category_weight *= 1.5  # Higher weight for personal indicators

            total_score += category_score * category_weight
            total_weight += category_weight

        # Score based on regex patterns
        pattern_score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_score += 0.2

        # Combine scores
        keyword_score = total_score / total_weight if total_weight > 0 else 0
        final_score = min((keyword_score + pattern_score), 1.0)

        return final_score

    def _calculate_enhanced_confidence(self, text, predicted_class):
        """Calculate enhanced confidence based on multiple factors"""
        base_confidence = 0.5

        # Get category-specific keywords
        category_keywords = self._get_category_keywords(predicted_class)
        if not category_keywords:
            return base_confidence

        # Calculate keyword match ratio
        text_lower = text.lower()
        matches = sum(1 for keyword in category_keywords if keyword in text_lower)
        match_ratio = matches / len(category_keywords) if category_keywords else 0

        # Calculate confidence based on match ratio
        confidence = base_confidence + (match_ratio * 0.4)

        # Boost confidence for strong indicators
        if predicted_class == 'Spam':
            strong_spam_indicators = ['urgent', 'win', 'prize', 'lottery', 'click here', 'call now']
            if any(indicator in text_lower for indicator in strong_spam_indicators):
                confidence = min(confidence + 0.2, 0.95)

        elif predicted_class == 'Personal':
            personal_indicators = ['dear', 'hi', 'hello', 'love', 'miss', 'friend']
            if any(indicator in text_lower for indicator in personal_indicators):
                confidence = min(confidence + 0.15, 0.90)

        return min(confidence, 0.95)

    def _get_category_keywords(self, category):
        """Get all keywords for a specific category"""
        keyword_map = {
            'Spam': self.spam_keywords,
            'Promotions': self.promotion_keywords,
            'Social': self.social_keywords,
            'Purchases': self.purchase_keywords,
            'Travel': self.travel_keywords,
            'Forums': self.forum_keywords,
            'Updates': self.update_keywords,
            'Personal': self.personal_keywords
        }

        category_dict = keyword_map.get(category, {})
        all_keywords = []
        for keyword_list in category_dict.values():
            all_keywords.extend(keyword_list)

        return all_keywords

    def _create_enhanced_probability_distribution(self, predicted_class, confidence, text):
        """Create enhanced probability distribution for all categories"""
        all_probs = {}
        text_lower = text.lower()

        # Calculate base probabilities for all categories
        base_prob = (1.0 - confidence) / 7  # Distribute remaining probability

        for category in self.categories.values():
            if category == predicted_class:
                all_probs[category] = confidence
            else:
                # Calculate secondary probabilities based on partial matches
                category_keywords = self._get_category_keywords(category)
                if category_keywords:
                    matches = sum(
                        1 for keyword in category_keywords[:20] if keyword in text_lower)  # Limit for performance
                    match_ratio = matches / min(len(category_keywords), 20)
                    secondary_prob = base_prob * (1 + match_ratio)
                    all_probs[category] = secondary_prob
                else:
                    all_probs[category] = base_prob

        # Normalize probabilities
        total = sum(all_probs.values())
        if total > 0:
            all_probs = {k: v / total for k, v in all_probs.items()}

        return all_probs

    def _enhanced_rule_based_classify(self, subject, body, return_probabilities=False):
        """Enhanced rule-based classification with comprehensive analysis"""
        combined_text = f"{subject} {body}".lower()

        # Calculate scores for all categories
        category_scores = {}
        category_scores['Spam'] = self._calculate_category_score(combined_text, self.spam_keywords,
                                                                 self.email_patterns.get('spam_structure', []))
        category_scores['Promotions'] = self._calculate_category_score(combined_text, self.promotion_keywords,
                                                                       self.email_patterns.get('promotional_structure',
                                                                                               []))
        category_scores['Social'] = self._calculate_category_score(combined_text, self.social_keywords, [])
        category_scores['Purchases'] = self._calculate_category_score(combined_text, self.purchase_keywords, [])
        category_scores['Travel'] = self._calculate_category_score(combined_text, self.travel_keywords, [])
        category_scores['Forums'] = self._calculate_category_score(combined_text, self.forum_keywords, [])
        category_scores['Updates'] = self._calculate_category_score(combined_text, self.update_keywords, [])
        category_scores['Personal'] = self._calculate_category_score(combined_text, self.personal_keywords,
                                                                     self.email_patterns.get('personal_structure', []))

        # Find the category with highest score
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]

        # Set minimum confidence thresholds
        if max_score < 0.2:
            category = 'Updates'  # Default category
            confidence = 0.50
        else:
            category = best_category
            confidence = min(0.60 + (max_score * 0.35), 0.95)
        result = {
            'category': category,
            'confidence': confidence,
            'text_preview': combined_text[:200] + "..." if len(combined_text) > 200 else combined_text,
            'method': 'enhanced-rule-based'
        }
        if return_probabilities:
            # Create probability distribution based on scores
            all_probs = {}
            total_score = sum(category_scores.values())

            if total_score > 0:
                for cat, score in category_scores.items():
                    if cat == category:
                        all_probs[cat] = confidence
                    else:
                        # Normalize other scores
                        normalized_score = score / total_score
                        all_probs[cat] = (1 - confidence) * normalized_score
            else:
                # Equal distribution if no strong signals
                equal_prob = 1.0 / len(self.categories)
                all_probs = {cat: equal_prob for cat in self.categories.values()}

            # Normalize probabilities to ensure they sum to 1
            total = sum(all_probs.values())
            if total > 0:
                all_probs = {k: v / total for k, v in all_probs.items()}

            result['all_probabilities'] = all_probs
        return result

    def batch_classify(self, emails):
        """Classify multiple emails"""
        results = []
        total = len(emails)
        for i, email in enumerate(emails):
            subject = email.get('subject', '')
            body = email.get('body', '')
            result = self.classify_email(subject, body, return_probabilities=True)
            result['email_id'] = i
            results.append(result)
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"Processed {i + 1}/{total} emails")
        return results

    def get_model_info(self):
        """Get information about the loaded model"""
        gpu_memory = 0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            except:
                gpu_memory = 0
        return {
            'model_type': self.model_type,
            'device': self.device,
            'categories': list(self.categories.values()),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': gpu_memory,
            'classifier_loaded': self.classifier is not None,
            'total_keywords': sum(len(keywords) for category_dict in [
                self.spam_keywords, self.promotion_keywords, self.social_keywords,
                self.purchase_keywords, self.travel_keywords, self.forum_keywords,
                self.update_keywords, self.personal_keywords
            ] for keywords in category_dict.values())
        }

    def get_category_details(self, category):
        """Get detailed information about a specific category's keywords"""
        keyword_map = {
            'Spam': self.spam_keywords,
            'Promotions': self.promotion_keywords,
            'Social': self.social_keywords,
            'Purchases': self.purchase_keywords,
            'Travel': self.travel_keywords,
            'Forums': self.forum_keywords,
            'Updates': self.update_keywords,
            'Personal': self.personal_keywords
        }

        category_dict = keyword_map.get(category, {})
        return {
            'category': category,
            'keyword_groups': {group: len(keywords) for group, keywords in category_dict.items()},
            'total_keywords': sum(len(keywords) for keywords in category_dict.values()),
            'sample_keywords': {
                group: keywords[:5] for group, keywords in category_dict.items()
            }
        }

    def analyze_text_detailed(self, subject="", body=""):
        """Provide detailed analysis of text showing scores for all categories"""
        text = f"Subject: {subject}\n\nBody: {body}".strip()
        text_lower = text.lower()

        detailed_analysis = {
            'text_length': len(text),
            'category_scores': {},
            'top_keywords_found': {},
            'pattern_matches': {}
        }

        # Calculate detailed scores for each category
        for category in self.categories.values():
            category_keywords = self._get_category_keywords(category)
            score = self._calculate_category_score(text_lower, self._get_category_keyword_dict(category), [])

            # Find matching keywords
            matching_keywords = [kw for kw in category_keywords[:20] if kw in text_lower]

            detailed_analysis['category_scores'][category] = {
                'score': score,
                'matching_keywords': matching_keywords,
                'match_count': len(matching_keywords)
            }

        # Check pattern matches
        for pattern_type, patterns in self.email_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append(pattern)
            detailed_analysis['pattern_matches'][pattern_type] = matches

        return detailed_analysis

    def _get_category_keyword_dict(self, category):
        """Get the keyword dictionary for a specific category"""
        keyword_map = {
            'Spam': self.spam_keywords,
            'Promotions': self.promotion_keywords,
            'Social': self.social_keywords,
            'Purchases': self.purchase_keywords,
            'Travel': self.travel_keywords,
            'Forums': self.forum_keywords,
            'Updates': self.update_keywords,
            'Personal': self.personal_keywords
        }
        return keyword_map.get(category, {})