## Prepare data

For tsv data:

```bash
sed 's/\t/|||/g' infile > outfile
```

For json data:
```bash
jq -r '. | .question + " in ||| " + .program' infile > outfile
```
or also drop braces and single quotes:
```bash
jq -r '. | [.question, .program] | @tsv' infile | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/( \(| \)|\047)/,"",$2);print}' > outfile

jq -r '. | [.question, .program] | @tsv' train_template.json | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/( \(| \)|\047)/,"",$2);print}' > train_template_alignment_inp.txt
jq -r '. | [.question, .program] | @tsv' train_len.json | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/( \(| \)|\047)/,"",$2);print}' > train_len_alignment_inp.txt

jq -r '. | [.question, .program] | @tsv' train_small.json | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/( \(| \)|\047)/,"",$2);print}' > train_small_alignment_inp.txt
```
or in addition drop the first `answer` in target sequence:
```bash
jq -r '. | [.question, .program] | @tsv' infile | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/(\( | \)|\047|^answer )/,"",$2);print}' > outfile

jq -r '. | [.question, .program] | @tsv' train_template.json | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/(\( | \)|\047|^answer )/,"",$2);print}' > train_template_alignment_inp.txt
jq -r '. | [.question, .program] | @tsv' train_len.json | awk 'BEGIN{FS="\t";OFS=" ||| "} {gsub(/(\( | \)|\047|^answer )/,"",$2);print}' > train_len_alignment_inp.txt
```

## Patch

Change `what's` to `what 's`.

## Run alignment

Go to `vendor/efmaral` and run
```bash
python align.py -m 1 -i infile > outfile
python align.py -m 3 -i ../../data/geo/funql/train_template_alignment_inp.txt --output-prob ../../data/geo/funql/train_template_alignment_oup.3.pkl
python align.py -m 3 -i ../../data/geo/funql/train_template_alignment_inp.txt > ../../data/geo/funql/train_template_alignment_oup.3.txt
python align.py -m 3 -i ../../data/geo/funql/train_len_alignment_inp.txt --output-prob ../../data/geo/funql/train_len_alignment_oup.3.pkl
python align.py -m 3 -i ../../data/geo/funql/train_len_alignment_inp.txt > ../../data/geo/funql/train_len_alignment_oup.3.txt

python align.py -m 3 -i ../../data/clevr/dsl/train_small_alignment_inp.txt > ../../data/clevr/dsl/train_small_alignment_oup.3.txt
```
