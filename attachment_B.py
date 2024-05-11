def tokenizer(url):
  tokens = re.split('[/-]', url)
  for i in tokens:
    if i.find(".") >= 0:
      dot_split = i.split('.')
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")
      tokens += dot_split
  return tokens
cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['URLs']
tVec = TfidfVectorizer(tokenizer=tokenizer) 
tfidf_X = tVec.fit_transform(train_df['URLs'])
