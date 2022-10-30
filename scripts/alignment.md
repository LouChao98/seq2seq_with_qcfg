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
```

## Run alignment

Go to `vendor/efmaral` and run
```bash
python align.py -m 1 -i infile > outfile
python align.py -m 3 -i ../../data/geo/funql/train_template_alignment_inp.txt --output-prob ../../data/geo/funql/train_template_alignment_oup.3.pkl > ../../data/geo/funql/train_template_alignment_oup.3.txt
python align.py -m 3 -i ../../data/geo/funql/train_len_alignment_inp.txt --output-prob ../../data/geo/funql/train_len_alignment_oup.3.pkl > ../../data/geo/funql/train_len_alignment_oup.3.txt
```
