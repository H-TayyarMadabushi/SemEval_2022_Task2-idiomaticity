import sys
import csv
from sklearn.metrics import f1_score

def load_csv( path ) : 
  header = None
  data   = list()
  with open( path, encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile ) 
    for row in reader : 
      if header is None : 
        header = row
        continue
      data.append( row ) 
  return header, data



def _score( submission_data, submission_headers, gold_data, gold_headers, languages, settings )  :

  if len( settings ) == 2 :
    assert len( languages ) == 2
    gold_data = gold_data + gold_data
    
  filtered_submission_data = [ i for i in submission_data if i[ submission_headers.index( 'Language' ) ] in languages and i[ submission_headers.index( 'Setting' ) ] in settings ]
  submission_ids           = [ int( i[ submission_headers.index( 'ID' ) ] ) for i in filtered_submission_data  ] 
  
  filtered_gold_data       = [ i for i in gold_data       if int( i[ gold_headers.index( 'ID' ) ] ) in submission_ids ]
  
  gold_ids       = [ int( i[0] ) for i in filtered_gold_data ]
  
  if submission_ids != gold_ids :
    print( "ERROR: IDs in Submission file do not match IDs in gold file!" )
    sys.exit()

  y_pred = [ i[ submission_headers.index( 'Label' ) ] for i in filtered_submission_data ]
  y_true = [ i[ gold_headers      .index( 'Label' ) ] for i in filtered_gold_data       ]

  if any( [ ( i == '' ) for i in y_pred ] ) : 
    return None, None, None

  y_pred = [ int( i ) for i in y_pred ]
  y_true = [ int( i ) for i in y_true ]
  f1_weighted = f1_score( y_true, y_pred, average='weighted' )
  f1_micro    = f1_score( y_true, y_pred, average='micro'    )
  f1_binary   = f1_score( y_true, y_pred, average='binary'   )

  return f1_weighted, f1_micro, f1_binary


def evaluate_submission( submission_file, gold_labels ) :
  
  submission_headers, submission_data = load_csv( submission_file ) 
  gold_headers      , gold_data       = load_csv( gold_labels     ) 

  if submission_headers != ['ID', 'Language', 'Setting', 'Label'] : 
    print( "ERROR: Incorrect submission format", file=sys.stderr ) 
    sys.exit()
  
  if gold_headers       != ['ID', 'DataID', 'Language', 'Label'] : 
    print( "ERROR: Incorrect gold labels data format (did you use the correct file?)", file=sys.stderr ) 
    sys.exit()

  output = [ [ 'Settings', 'Languages', "F1 Score (weighted)", "F1 Score (micro)", "F1 Score (binary)" ] ]
  
  for languages, settings in [ 
     [ [ 'EN' ]      , [ 'zero_shot' ] ], 
     [ [ 'PT' ]      , [ 'zero_shot' ] ], 
     [ [ 'EN', 'PT' ], [ 'zero_shot' ] ], 

     [ [ 'EN' ]      , [ 'one_shot'  ] ], 
     [ [ 'PT' ]      , [ 'one_shot'  ] ], 
     [ [ 'EN', 'PT' ], [ 'one_shot'  ] ], 

    ] : 
    f1_weighted, f1_micro, f1_binary = _score( submission_data, submission_headers, gold_data, gold_headers, languages, settings ) 
    this_entry         = [ ','.join( settings ), ','.join( languages), f1_weighted, f1_micro, f1_binary ]
    output.append( this_entry ) 

  return output



if __name__ == '__main__' :
  
  submission_file = sys.argv[1]
  gold_file       = sys.argv[2]

  
  results = evaluate_submission( submission_file, gold_file )

  for row in results :
    print( '\t'.join( [ str( i ) for i in row ] ) ) 
