class FetchManager:
    def __init__(self, sess, fetches):
        # fetches is a tuple
        self.fetches = fetches
        # sess is a tf session
        self.sess = sess

    def fetch(self, feed_dictionary, additional_fetches=[]):
        fetches = self.fetches + additional_fetches
        evaluation = self.sess.run(fetches, feed_dictionary)
        return {key: value for key, value in zip(fetches, evaluation)}