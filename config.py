class Config(object):

    DEBUG = False
    TESTING = False

    DB_NAME = ''
    DB_USERNAME = ''
    DB_PASSWORD = ''
    UPLOADS = ''


class ProductionConfig(Config):
    DEBUG = False
    UPLOADS = ''


class DevelopmentConfig(Config):
    DEBUG = True
    UPLOADS = ''


class TestingConfig(Config):
    DEBUG = True
    UPLOADS = ''
