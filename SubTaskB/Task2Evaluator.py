import csv
from scipy.stats import pearsonr, spearmanr

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

  # ['ID', 'Language', 'Setting', 'Sim']
  # ['ID', 'DataID', 'Language', 'sim', 'otherID']

  filtered_submission_data = [ i for i in submission_data if i[ submission_headers.index( 'Language' ) ] in languages and i[ submission_headers.index( 'Setting' ) ] in settings ]
  if any( [ ( i[ submission_headers.index( 'Sim' ) ] == '' ) for i in filtered_submission_data ] ) : 
    return None, None, None
  
  filtered_submission_dict = dict()
  for elem in filtered_submission_data : 
    filtered_submission_dict[ elem[ submission_headers.index( 'ID' ) ] ] = elem[ submission_headers.index( 'Sim' ) ]

  ## Generate gold 
  if len( settings ) > 1 : 
    gold_data = gold_data + gold_data
    assert len( languages ) == 2 
  else : 
    gold_data = [ i for i in gold_data if i[ gold_headers.index( 'Language' ) ] in languages ]

  gold_labels_all = list()
  predictions_all = list()

  gold_labels_sts = list()
  predictions_sts = list()

  gold_labels_no_sts = list()
  predictions_no_sts = list()

  for elem in gold_data :
    this_sim = elem[ gold_headers.index( 'sim' ) ]
    if this_sim == '' : 
      this_sim = filtered_submission_dict[ elem[ gold_headers.index( 'otherID' ) ] ]
    this_sim = float( this_sim )
    this_prediction = float( filtered_submission_dict[ elem[ gold_headers.index( 'ID' ) ] ] )

    gold_labels_all.append( this_sim        )
    predictions_all.append( this_prediction )

    if elem[ gold_headers.index( 'DataID' ) ].split( '.' )[2] == 'sts' :
      gold_labels_sts.append( this_sim        )
      predictions_sts.append( this_prediction )
    else : 
      gold_labels_no_sts.append( this_sim        )
      predictions_no_sts.append( this_prediction )


  corel_all, pvalue    =  spearmanr(gold_labels_all   , predictions_all    ) 
  corel_sts, pvalue    =  spearmanr(gold_labels_sts   , predictions_sts    ) 
  corel_no_sts, pvalue =  spearmanr(gold_labels_no_sts, predictions_no_sts ) 
  return ( corel_all, corel_sts, corel_no_sts )

def evaluate_submission( submission_file, gold_labels ) : 
  submission_headers, submission_data = load_csv( submission_file ) 
  gold_headers      , gold_data       = load_csv( gold_labels     ) 

  if submission_headers != ['ID', 'Language', 'Setting', 'Sim'] : 
    print( "ERROR: Incorrect submission format", file=sys.stderr ) 
    sys.exit()
  if gold_headers !=   ['ID', 'DataID', 'Language', 'sim', 'otherID']: 
    print( "ERROR: Incorrect gold labels data format (did you use the correct file?)", file=sys.stderr ) 
    sys.exit()

  submission_ids = [ int( i[0] ) for i in submission_data  ] 
  gold_ids       = [ int( i[0] ) for i in gold_data        ] + [ int( i[-1] ) for i in gold_data if i[-1] != '' ] 

  for id in submission_ids : 
    if not id in gold_ids :  
      print( "ERROR: Submission file contains IDs that gold data does not - this could be because you submitted the wrong results (dev results instead of evaluation results) or because your submission file is corrupted", file=sys.stderr )
      sys.exit()
  
  output = [ [ 'Settings', 'Languages', "Spearman Rank ALL", "Spearman Rank Idiom Data", "Spearman Rank STS Data" ] ]
  for languages, settings in [ 
     [ [ 'EN' ]      , [ 'pre_train' ] ], 
     [ [ 'PT' ]      , [ 'pre_train' ] ], 
     [ [ 'EN', 'PT' ], [ 'pre_train' ] ], 

     [ [ 'EN' ]      , [ 'fine_tune' ] ], 
     [ [ 'PT' ]      , [ 'fine_tune' ] ], 
     [ [ 'EN', 'PT' ], [ 'fine_tune' ] ], 

     [ [ 'EN', 'PT' ], [ 'fine_tune', 'pre_train' ] ]
    ] : 
    corel_all, corel_sts, corel_no_sts = _score( submission_data, submission_headers, gold_data, gold_headers, languages, settings ) 
    this_entry         = [ ','.join( settings ), ','.join( languages), corel_all, corel_no_sts, corel_sts ] 
    output.append( this_entry ) 

  return output

