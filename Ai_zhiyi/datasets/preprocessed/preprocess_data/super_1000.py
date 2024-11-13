import config
from sklearn import preprocessing
import joblib
from collections import defaultdict
import operator

# 配置数据
questions_train = config.questions_train
answers_train = config.answers_train
images_train = config.images_train

# 设定最多选取多少个回答
max_answers = 1000
answer_fq = defaultdict(int)

class NewData:
    def __init__(self, questions_train, answers_train, images_train):
        self.questions_train = questions_train
        self.answers_train = answers_train
        self.images_train = images_train

    def get_data(self):
        # 统计每个答案的频率
        for answer in self.answers_train:
            answer_fq[answer] += 1
        
        # 按照出现次数排序，选择前1000个答案
        sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[:max_answers]
        top_answers, _ = zip(*sorted_fq)
        
        new_answers_train = []
        new_questions_train = []
        new_images_train = []
        
        # 只保留前1000个答案相关的数据
        for answer, question, image in zip(self.answers_train, self.questions_train, self.images_train):
            if answer in top_answers:
                new_answers_train.append(answer)
                new_questions_train.append(question)
                new_images_train.append(image)
        
        return new_questions_train, new_answers_train, new_images_train

# 实例化类并获取数据
new_data_instance = NewData(questions_train, answers_train, images_train)
questions_train, answers_train, images_train = new_data_instance.get_data()

# 对答案进行编号
labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
joblib.dump(labelencoder, 'D:\Ai_zhiyi\datasets\preprocessed\date\labelencoder.pkl')

# 打印一些示例数据
print("前几个问题:", questions_train[:9])
print("前几个答案:", answers_train[:9])
print("前几张图片:", images_train[:9])