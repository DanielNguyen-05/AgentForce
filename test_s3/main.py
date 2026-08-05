import boto3

bucket = "agent-force-video-bucket"
prefix = "dataset/keyframes/L21/"

s3 = boto3.client("s3")

response = s3.list_objects_v2(
    Bucket=bucket,
    Prefix=prefix
)

for obj in response.get("Contents", []):
    print(obj["Key"])