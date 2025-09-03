# Universal Media Authenticator - Ready to Use!

## Sign any media file:
python3 external_signer.py your_video.mp4 example_private.json

## Verify any signed file:  
python3 external_verifier.py your_video.mp4 your_video.mp4.qrmn example_public.json

## Supported formats:
- Video: .mp4, .avi, .mov, .mkv, .webm
- Audio: .wav, .mp3, .flac, .m4a, .ogg
- Works with ANY complexity level!

## Results:
- GREEN: Authentic and verified
- AMBER: Likely authentic with minor issues  
- RED: Not authentic or tampered

## Files created:
- [filename].qrmn = signature file (keep with media)
- Contains quantum-resistant signatures
- Detects any file tampering instantly

