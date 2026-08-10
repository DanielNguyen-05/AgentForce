resource "google_storage_bucket" "video_bucket" {
  name     = "agent-force-video-bucket"
  location = "asia-southeast1"

  versioning {
    enabled = true
  }

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}
