import tweepy
import re

# Masukkan kredensial API Twitter Anda di sini
consumer_key = "2fAzjiVkfk0TWFDpQAH0Ui2lE"
consumer_secret = "TuXhKRVeWc68iOTpfiWEa2MxoN6glvRloJ4OJQ3ReXUsGaR4Qi"
access_token = "1797229045434007552-hDApRKRa3moM34Ave5p05z92F6cqyf"
access_token_secret = "HeagbbcOXtfAzOqqI3hYrXvaOnYeg0riX0BrWsCgGqAne"

# Autentikasi ke API Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Buat instance API
api = tweepy.API(auth)

def get_tweet_content(url):
    # Ekstrak ID tweet dari URL
    match = re.search(r'/status/(\d+)', url)
    if not match:
        return "URL tidak valid"
    
    tweet_id = match.group(1)
    
    try:
        # Ambil tweet
        tweet = api.get_status(tweet_id, tweet_mode='extended')
        return tweet.full_text
    except tweepy.TweepError as e:
        return f"Error: {str(e)}"

# Loop utama
while True:
    url = input("Masukkan URL tweet (atau 'q' untuk keluar): ")
    if url.lower() == 'q':
        break
    
    content = get_tweet_content(url)
    print(f"\nIsi tweet:\n{content}\n")

print("Terima kasih telah menggunakan Twitter Content Extractor!")