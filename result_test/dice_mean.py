import numpy as np

path = r'/home/dkuang/Github/Medical-image-registration/result_test/'
dice_before = np.load(path+'dice_before.npy')
dice_after = np.load(path+'dice_after.npy')


dice_before_mean = np.mean(dice_before, 0)
dice_after_mean = np.mean(dice_after, 0)

dice_before_median = np.median(dice_before, 0)
dice_after_median = np.median(dice_after, 0)


count_worse=0
count_equal = 0
count_better = 0

for i in range(56):

    if dice_after_mean[i] < dice_before_mean[i]:

        count_worse += 1

    elif dice_after_mean[i] > dice_before_mean[i]:

        count_better += 1

    else:

        count_equal += 1

print('worse(mean): {}'.format(count_worse))
print('equal(mean): {}'.format(count_equal))
print('better(mean): {}'.format(count_better))


count_worse=0
count_equal = 0
count_better = 0

for i in range(56):

    if dice_after_median[i] < dice_before_median[i]:

        count_worse += 1

    elif dice_after_median[i] > dice_before_median[i]:

        count_better += 1

    else:

        count_equal += 1

print('worse(median): {}'.format(count_worse))
print('equal(median): {}'.format(count_equal))
print('better(median): {}'.format(count_better))

