#import os

# To generate a new secret key:
# >>> import random, string
# >>> "".join([random.choice(string.printable) for _ in range(24)])

#basedir = os.path.abspath(os.path.dirname(__file__))
#SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')

SOURCE_FILE = 'OC_DS_P6_prod.pkl'

REGEX = '[a-z0-9]+[#-]?\+{0,2}[a-z0-9]*'

# Extra stopwords = radicaux qui ne me semblent pas discriminants
EXTRA_SW = ('use', 'get', 'like', 'way', 'creat', 'would', 'want', 'need',\
            'know', 'could', 'x', 'xx', 'xyz', 'aa', 'xxx', 'z', 'yyyi', 'wont',\
            'aaa', 'aaaaaa', 'aabbc', 'aandb', 'aarrggbb', 'without', 'behind', \
            'within')
