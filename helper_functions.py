import nltk, tqdm, sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tqdm.notebook import tqdm
from absl import flags
from app import sess, graph
import string

sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)

MAX_WORD_COUNT = 70

id_to_pos = {}
pos_to_id = {}
with open('ID_POS.txt', 'r') as f:
    for line in f:
        pos_id, pos = line.strip().split()
        pos_id = int(pos_id)
        id_to_pos[pos_id] = pos
        pos_to_id[pos] = pos_id
        
class BatchPaddingExample:
    """
    Được dùng để bổ sung ví dụ giả cho batch cho đến khi số lượng ví dụ trở thành bội số của batch size. 
    Vì khi huấn luyện theo từng batch thì có thể có batch (như batch cuối) sẽ không đủ số lượng ví dụ nên có thể gây sai lệch.
    """

class TrainingExample:
    def __init__(self, input_id, text_input1, text_input2 = None, label = None):
        """
            id (str): Định danh của ví dụ.
            text_input1 (str): Chuỗi văn bản đầu tiên (bắt buộc).
            text_input2 (str, optional): Chuỗi văn bản thứ hai (tùy chọn).
            label (str, optional): Nhãn của ví dụ (tùy chọn).
        """
        self.id = input_id
        self.text_input1 = text_input1
        self.text_input2 = text_input2
        self.label = label

# Chuyển đổi các văn bản thành các đối tượng TrainingExample
def create_examples_from_text(texts, labels):

    # Duyệt qua từng text và label và tạo một TrainingExample tương ứng
    examples = [TrainingExample(input_id=None, text_input1=text, text_input2=None, label=label) \
                for text, label in zip(texts, labels)]
    return examples

# Cắt ngắn các tokens để độ dài không vượt quá max_length
def truncate_tokens(tokens, max_length):
    if len(tokens) > max_length - 2:
        tokens = tokens[: max_length - 2]
    return tokens

# Thêm các padding (giá trị 0) để cho độ dài của sequence bằng được max_length
def pad_sequence(sequence, max_length, padding_value=0):
    while len(sequence) < max_length:
        sequence.append(padding_value)
    return sequence

# Chuyển đổi một example thành các đặc trưng input_ids, input_mask, segment_ids và label_ids
def example_to_feature(tokenizer, example, max_length=256):
    
    # Nếu example là BatchPaddingExample thì trả về toàn bộ là số 0
    if isinstance(example, BatchPaddingExample):
        input_ids, input_mask, segment_ids, label_ids = [0] * 4 * max_length
        return input_ids, input_mask, segment_ids, label_ids
    
    '''
        original_tokens: Danh sách các token ban đầu của example.
        token_to_original_map: Mapping vị trí của các token BERT trong chuỗi đầu ra với các token ban đầu trong chuỗi đầu vào.
        bert_tokens: Danh sách các BERT token được tạo ra từ các token ban đầu.
    '''
    
    original_tokens = truncate_tokens(example.text_input1, max_length)
    token_to_original_map, bert_tokens, segment_ids = [], [], []

    bert_tokens.append("[CLS]")
    segment_ids.append(0)
    token_to_original_map.append(len(bert_tokens)-1)
    
    # Chuyển đổi các tokens ban đầu thành các tokens của BERT và lưu lại mapping vị trí
    for token in original_tokens:
        bert_tokens.extend(tokenizer.tokenize(token))
        token_to_original_map.append(len(bert_tokens) - 1)
        segment_ids.append(0)

    bert_tokens.append("[SEP]")
    segment_ids.append(0)
    token_to_original_map.append(len(bert_tokens)-1)
    input_ids = tokenizer.convert_tokens_to_ids([bert_tokens[i] for i in token_to_original_map])
    
    # Tạo input_mask và label_ids từ input_ids và label của ví dụ
    input_mask = [1] * len(input_ids)
    label_ids = [0] + [pos_to_id[label] for label in example.label] + [0]

    # Thêm padding để độ dài của sequence bằng max_length
    [pad_sequence(seq, max_length) for seq in [input_ids, input_mask, segment_ids, label_ids]]

    return input_ids, input_mask, segment_ids, label_ids

# Chuyển đổi các example thành các ma trận đặc trưng đầu vào cho mô hình BERT
def create_features_from_examples(tokenizer, examples, max_length=256):
    input_ids, input_masks, segment_ids, labels = [], [], [], []
    
    # Duyệt qua từng example và chuyển đổi thành feature
    for example in tqdm(examples, desc="Converting examples to features", disable=True):
        input_id, input_mask, segment_id, label = example_to_feature(tokenizer, example, max_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
        
    # Trả về input_ids, input_masks, segment_ids, labels dưới dạng numpy array
    return (np.array(input_ids), np.array(input_masks), np.array(segment_ids), np.array(labels))

# Tải tokenizer của model BERT đã được đào tạo
def load_bert_tokenizer():
    bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
    
    # Lấy thông tin về việc mã hóa văn bản từ module BERT
    token_info = bert_module(signature = "tokenization_info", as_dict = True)
    vocab_file_path, lower_case = sess.run([token_info["vocab_file"], token_info["do_lower_case"],])
    
    # Tạo BERT tokenizer
    bert_tokenizer = FullTokenizer(vocab_file = vocab_file_path, do_lower_case = lower_case)
    
    return bert_tokenizer

# Tạo tokenizer để chuyển đổi văn bản đầu vào thành một chuỗi các token:
tokenizer = load_bert_tokenizer()

class BERTLayer(Layer):
    def __init__(self, output_representation='sequence_output', trainable=True, **kwargs):
        """
            output_representation: sequence_output: đầu ra là chuỗi các vector biểu diễn của các từ trong câu. 
                                   pooled_output  : đầu ra là một vector biểu diễn của toàn bộ câu văn bản.
            trainable: các trọng số của mô hình BERT có được huấn luyện lại hay không.
        """
        super(BERTLayer, self).__init__(**kwargs)

        self.trainable = trainable
        self.output_representation = output_representation
        self.bert = None

    # Tạo các trọng số
    def build(self, input_shape):
        self.bert = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
                               trainable=self.trainable, 
                               name="{}_module".format(self.name))

        # Loại bỏ các trọng số có tên chứa chuỗi con "/cls/" hoặc "/pooler/"
        removed_str = ["/cls/", "/pooler/"]
        self._trainable_weights += [var for var in self.bert.variables[:] if not any(x in var.name for x in removed_str)]
            
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
            
        super(BERTLayer, self).build(input_shape)

    # Xử lý các đặc trưng và trả về 1 tensor đại diện cho biểu diễn đầu ra của mô hình BERT
    def call(self, inputs, mask=None):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[self.output_representation]
        return result
    
    # Trả về tensor kích thức như input_ids và kiểm tra input_ids có khác 0 hay không
    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs[0], 0.0)   

    # Tính toán kích thước đầu ra của lớp
    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 768)
        else:
            return (None, None, 768)

def build_model(max_length):
    # Thiết lập hạt giống cho TensorFlow
    seed = 0 

    # Định nghĩa đầu vào
    ids =tf.keras.layers.Input(shape=(max_length,), name="input_ids")
    masks = tf.keras.layers.Input(shape=(max_length,), name="input_masks")
    segments = tf.keras.layers.Input(shape=(max_length,), name="segment_ids")
    bert_inputs = [ids, masks, segments]

    # Tạo mô hình BERT và lấy đầu ra
    np.random.seed(seed)
    bert_output = BERTLayer()(bert_inputs)

    # Thêm lớp Dense để phân loại
    np.random.seed(seed)
    outputs = tf.keras.layers.Dense(len(pos_to_id), activation=tf.keras.activations.softmax)(bert_output)

    # Thiết lập tối ưu hóa, hàm mất mát và các độ đo
    np.random.seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00004)
    loss_fn = tf.keras.losses.categorical_crossentropy
    metrics = ['accuracy']

    # Định nghĩa và biên dịch mô hình
    np.random.seed(seed)
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)   

    return model
      
def custom_predict(self, text):
    cased_sentence = nltk.word_tokenize(text)
    uncased_sentence = nltk.word_tokenize(text.lower())

    test_example = create_examples_from_text([uncased_sentence], [['-PAD-']*len(uncased_sentence)])
    (input_ids, input_masks, segment_ids, _) = create_features_from_examples(tokenizer, test_example, max_length=MAX_WORD_COUNT+2)

    # Gọi hàm predict của lớp cha
    with sess.as_default():
        with graph.as_default():
            prediction = self.predict([input_ids, input_masks, segment_ids], batch_size=1).argmax(-1)[0]              

    predict = []
    for i, pred in enumerate(prediction):
        if pred!=0 and i <= len(cased_sentence):
            predict.append([cased_sentence[i-1],id_to_pos[pred]])
    return predict    

def split_sentence(sentence):
    words = sentence.split()
    new_words = []
    for word in words:
        # Check if the last character of the word is a punctuation mark
        if word[-1] in string.punctuation:
            # Split the punctuation mark from the word and add it as a separate word
            new_words.append(word[:-1])
            new_words.append(word[-1])
        else:
            new_words.append(word)
    # Split the new list of words into sentences with a maximum word count of max_words
    sentences = []
    current_sentence = ""
    for word in new_words:
        if len(current_sentence.split()) < MAX_WORD_COUNT:
            current_sentence += word + " "
        else:
            sentences.append(current_sentence.strip())
            current_sentence = word + " "
    if current_sentence:
        sentences.append(current_sentence.strip())
    return sentences