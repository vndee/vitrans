### Speech-To-Speech Module

To training speech-to-speech system, we must align the data into json lines format as follow:


```json
{ "translation": { "en": "Suddenly it looks bigger", "vi": "Đột nhiên nó trông lớn hơn" } }
{ "translation": { "en": "You cant abandon Tom", "vi": "bạn không thể từ bỏ tom" } }
{ "translation": { "en": "Youd be surprised what you can learn in a week", "vi": "bạn sẽ ngạc nhiên về những gì bạn có thể học trong một tuần" } }
{ "translation": { "en": "The cat is sleeping on the chair", "vi": "con mèo đang ngủ trên ghế" } }
{ "translation": { "en": "The area around the melting ice cream was crawling with ants", "vi": "khu vực xung quanh kem tan chảy đang bò với kiến" } }
{ "translation": { "en": "They enjoy one anothers company", "vi": "họ thích công ty của nhau" } }
{ "translation": { "en": "It seems less crowded during the week", "vi": "Có vẻ như ít đông đúc hơn trong tuần" } }
{ "translation": { "en": "Both Nancy and Jane were absent from school", "vi": "cả nancy và jane đều vắng mặt ở trường" } }
{ "translation": { "en": "Im a coward when it comes to cockroaches", "vi": "Tôi là một kẻ hèn nhát khi nói đến gián" } }
{ "translation": { "en": "Thatll be interesting", "vi": "Điều đó sẽ rất thú vị" } }
```

Then run the training script:
```bash
sh train.sh
```
Or you can modify the parameters:
```bash
python train.py \
    --model_name_or_path facebook/mbart-large-50-one-to-many-mmt \
    --do_train \
    --do_eval \
	--do_predict \
    --source_lang en \
    --target_lang vi \
    --train_file data/kaggle/train.json \
    --test_file data/kaggle/test.json \
    --validation_file data/kaggle/val.json \
    --output_dir ./output/tst-translation \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --predict_with_generate
```
