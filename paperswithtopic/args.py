import argparse

def list2dict(args):
    
    _tuples = []
    for a in args:
        
        k, v = a.split('=')
        try: # IF IT'S A NUMBER
            if v.isdigit(): # TRY INTEGER FIRST
                v = int(v)
            else: # AND FLOAT HERE.
                v = float(v) # ERROR WILL OCCUR IF NON-CASTABLE STRING GIVEN
            
        except: # JUST LET v STRING
            pass
        
        _tuples.append((k, v))
        
    return {k: v for (k, v) in _tuples}


def parse_args():

    parser = argparse.ArgumentParser()
    _, unknownargs  = parser.parse_known_args()

    return list2dict(unknownargs)


if __name__=="__main__":

    args = parse_args()
    print(args)