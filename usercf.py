import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict
import csv

random.seed(0)


class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = 20
        self.n_rec_movie = 10

        self.user_sim_mat = {}
        self.movie_count = 0

        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print ('recommended movie number = %d' %
               self.n_rec_movie, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7,dataset='douban'):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        with open(filename,'r',encoding='utf-8') as f:
            if dataset=='douban':
                reader = csv.reader(f, skipinitialspace=True)
                reader.__next__() # Skip first line
                for paramList in reader:
                    #评分,用户名,评论时间,用户ID,电影名,类型
                    rating, user, time, userId, movie, genre = paramList
                    # split the data by pivot
                    if random.random() < pivot:
                        self.trainset.setdefault(user, {})
                        self.trainset[user][movie] = int(rating)
                        trainset_len += 1
                    else:
                        self.testset.setdefault(user, {})
                        self.testset[user][movie] = int(rating)
                        testset_len += 1                    
            elif dataset=='movielens':
                for line in self.loadfile(filename):
                    user, movie, rating, _ = line.split('::')
                    # split the data by pivot
                    if random.random() < pivot:
                        self.trainset.setdefault(user, {})
                        self.trainset[user][movie] = int(rating)
                        trainset_len += 1
                    else:
                        self.testset.setdefault(user, {})
                        self.testset[user][movie] = int(rating)
                        testset_len += 1            
            else:
                pass


        user_num=len(self.testset)+len(self.trainset)

        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)
        print ('user number = %s' % user_num, file=sys.stderr)
        print ('data number = %s' % (trainset_len+testset_len), file=sys.stderr)
    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
        print ('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
        print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print ('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print ('build user co-rated movies matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.trainset[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f'%(precision, recall, coverage), file=sys.stderr)


if __name__ == '__main__':
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    usercf = UserBasedCF()
    usercf.generate_dataset('user.csv',dataset='douban')
    usercf.calc_user_sim()
    usercf.evaluate()

    usercfm = UserBasedCF()
    usercfm.generate_dataset(ratingfile,dataset='movielens')
    usercfm.calc_user_sim()
    usercfm.evaluate()