pattern = r"((?:" + '|'.join(numbers) + r"|\d)(?:\s(?:" + '|'.join(numbers) + r"|\d|and))*)(.*?)(\d+[\.|\,]?\d*)\b\s*(\$|dollar)"
