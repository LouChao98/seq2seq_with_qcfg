# mainly from `span-based-sp`

import re


class Executor:
    def execute(self, program: str, kb_str: str = None) -> str:
        raise NotImplementedError

    def recovery(self, prediction) -> str:
        raise NotImplementedError


class ProgramExecutorGeo(Executor):
    # fmt: off
    predicate = {
        "cityid", "countryid", "placeid", "riverid", "stateid", "capital", "city", "lake", "mountain", "place", "river",
        "state", "major", "area_1", "capital_1", "capital_2", "density_1", "elevation_1", "elevation_2", "high_point_1",
        "high_point_2", "higher_2", "loc_1", "loc_2", "longer", "lower_2", "len", "next_to_1", "next_to_2",
        "population_1", "size", "traverse_1", "traverse_2", "answer", "largest", "largest_one", "smallest",
        "smallest_one", "highest", "lowest", "longest", "shortest", "count", "most", "fewest", "sum","exclude",
        "low_point_2", "low_point_1", 'intersection'
    }
    speical_tokens = {'all', '_', '0'}
    # fmt: on

    def __init__(self, geobase_pl, geoquery_pl, eval_pl):
        from pyswip import Prolog

        self._prolog = Prolog()
        self._prolog.consult(geobase_pl)
        self._prolog.consult(geoquery_pl)
        self._prolog.consult(eval_pl)

    def execute(self, program: str, kb_str: str = None) -> str:
        # make sure entities with multiple words are parsed correctly
        program = re.sub("' (\w+) (\w+) '", "'" + r"\1#\2" + "'", program)
        program = re.sub("' (\w+) (\w+) (\w+) '", "'" + r"\1#\2#\3" + "'", program)
        program = program.replace(" ", "").replace("#", " ")

        try:
            answers = list(self._prolog.query("eval(" + "{}, X".format(program) + ").", maxresult=1))
        except Exception as e:
            return "error_parse: {}".format(e)
        return str([str(answer) for answer in answers[0]["X"]])

    @classmethod
    def recovery(cls, prediction) -> str:
        result = []

        def _process(i):
            i += 1
            if i >= len(prediction):
                return i
            token = prediction[i]

            if token in cls.speical_tokens:
                result.append(token)
            elif token in cls.predicate:
                result.append(token)
                # binary predicate
                if token in ("cityid", "exclude", "intersection"):
                    result.append("(")
                    i = _process(i)
                    j = i + 1

                    if result[-1][0] != "'":
                        result.append(",")
                        i = _process(i)
                        result.append(")")
                        return i

                    if j < len(prediction):
                        next_token = prediction[j]
                        if next_token in cls.speical_tokens:
                            result.extend([",", next_token, ")"])
                            return j

                    rollback_token = result.pop()
                    parts = rollback_token.split()
                    if len(parts) == 1:
                        result.extend([rollback_token, ",", "_", ")"])
                    else:
                        result.extend([" ".join(parts[:-1]) + "'", ",", "'" + parts[-1], ")"])
                else:
                    result.append("(")
                    i = _process(i)
                    result.append(")")
            else:
                j = i + 1
                for j in range(i + 1, len(prediction) + 1):
                    if j == len(prediction):
                        break
                    token = prediction[j]
                    if token in cls.predicate or token in ("all", "_"):
                        break
                # hard code patch
                if prediction[i : i + 3] == ["salt", "lake", "city"]:
                    j = i + 3
                result.append("'" + " ".join(prediction[i:j]) + "'")
                i = j - 1

            return i

        _process(-1)
        return " ".join(result)


if __name__ == "__main__":
    pred = ["answer", "population_1", "cityid", "austin", "tx"]

    ex = ProgramExecutorGeo.recovery(pred)
    print(ex)
