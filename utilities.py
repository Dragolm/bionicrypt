def writer(text: str):
    with open("temp.txt", 'w') as f:
        f.write(text)
        f.write('\n')

def spacer():
    with open("temp.txt", 'w') as f:
        f.write('\n')
        f.write('\n')

def appender(text: str):
    with open("temp.txt", 'a') as f:
        f.write(text)
        f.write('\n')

def filewriter(text: str, filename: str):
    with open(str(filename+'.txt'), 'w') as f:
        f.write(text)
        f.write('\n')

def fileappender(text: str, filename: str):
    with open(str(filename+'.txt'), 'a') as f:
        f.write(text)
        f.write('\n')