# Files:
- full.txt - all (head, label, tail) relations. Omitted "price".
- test.txt - test set: only includes "likes" relation
- train_likes - training set including only "likes" relation
- valid_likes - validation set including only "likes" relation
- train_KG - training set including all relations
- valid_KG - validation set including all relations

# Inputs for KGAT (https://github.com/LunaBlack/KGAT-pytorch):
- KGAT_entity_list, KGAT_item_list, KGAT_relation_list, LGAT_user_list: mapping of objects with remapped ID
- KG_final - full.txt in terms of remapped ID
