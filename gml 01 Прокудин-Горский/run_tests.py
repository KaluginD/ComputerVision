#!/usr/bin/python3

from os import environ
from os.path import join
from sys import argv, exit


def run_single_test(data_dir, output_dir):
    from align import align
    from skimage.io import imread, imsave
    parts = open(join(data_dir, 'g_coord.csv')).read().rstrip('\n').split(',')
    g_coord = (int(parts[0]), int(parts[1]))
    img = imread(join(data_dir, 'img.png'), plugin='matplotlib')

    aligned_img, (b_row, b_col), (r_row, r_col) = align(img, g_coord)

    with open(join(output_dir, 'output.csv'), 'w') as fhandle:
        print('%d,%d,%d,%d' % (b_row, b_col, r_row, r_col), file=fhandle)

    imsave(join(output_dir, 'aligned_img.png'), aligned_img)


def check_test(output_dir, gt_dir):

    with open(join(output_dir, 'output.csv')) as fhandle:
        parts = fhandle.read().rstrip('\n').split(',')
        b_row, b_col, r_row, r_col = map(int, parts)

    with open(join(gt_dir, 'gt.csv')) as fhandle:
        parts = fhandle.read().rstrip('\n').split(',')
        coords = map(int, parts[1:])
        gt_b_row, gt_b_col, _, _, gt_r_row, gt_r_col, diff_max = coords

    diff = abs(b_row - gt_b_row) + abs(b_col - gt_b_col) + \
        abs(r_row - gt_r_row) + abs(r_col - gt_r_col)

    if diff > diff_max:
        return 'Wrong answer'
    return 'Ok'


def grade(results):
    ok_count = 0
    for result in results:
        if result['result'] == 'Ok':
            ok_count += 1
    total_count = len(results)
    description = '%02d/%02d' % (ok_count, total_count)
    mark = ok_count / total_count * 10
    return description, mark


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system, run on single input
        if len(argv) != 3:
            print('Usage: %s data_dir output_dir' % argv[0])
            exit(0)

        run_single_test(argv[1], argv[2])
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print('Usage: %s tests_dir' % argv[0])
            exit(0)

        from glob import glob
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_input'))):
            output_dir = sub('input$', 'output', input_dir)
            makedirs(output_dir, exist_ok=True)
            gt_dir = sub('input$', 'gt', input_dir)

            try:
                start = time()
                run_single_test(input_dir, output_dir)
                end = time()
                running_time = end - start
            except:
                result = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    result = check_test(output_dir, gt_dir)
                except:
                    result = 'Checker error'
                    traceback = format_exc()

            test_num = input_dir[-8:-6]
            if result == 'Runtime error' or result == 'Checker error':
                print(test_num, result, '\n', traceback)
                results.append({'result': result})
            else:
                print(test_num, '%.2fs' % running_time, result)
                results.append({
                    'time': running_time,
                    'result': result})

        description, mark = grade(results)
        print('Mark:', mark, description)
