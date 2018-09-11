def blah (filename, result_filename):
    with open(filename) as _file:
        with open(result_filename, "w") as res_file:
            for line in _file:
                parts = line.strip().split('\t')
                new_line = parts[0] + ' ' + parts[-1] + '\n'
                res_file.write(new_line)

blah('../eng.testa.iob', 'eng.testa.clean.iob')
blah('../eng.testb.iob', 'eng.testb.clean.iob')
blah('../eng.train.iob', 'eng.train.clean.iob')
