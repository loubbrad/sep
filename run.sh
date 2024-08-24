python /home/loubb/work/mvsep/inference.py \
    --model_type mdx23c \
    --config_path /home/loubb/work/mvsep/piano/config.yaml \
    --start_check_point /home/loubb/work/mvsep/piano/model_ep_96_sdr_11.2103.ckpt \
    --input_folder /mnt/ssd1/data/mp3/raw/aria-cl/piano-concerto \
    --store_dir /mnt/ssd1/data/mp3/raw/aria-cl/piano-concerto-sep

python /home/loubb/work/mvsep/inference.py \
    --model_type mdx23c \
    --config_path /home/loubb/work/mvsep/piano/config.yaml \
    --start_check_point /home/loubb/work/mvsep/piano/model_ep_96_sdr_11.2103.ckpt \
    --input_folder /mnt/ssd1/data/mp3/raw/aria-cl/jazz-trio \
    --store_dir /mnt/ssd1/data/mp3/raw/aria-cl/jazz-trio-sep


