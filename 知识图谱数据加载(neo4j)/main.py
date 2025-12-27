import json
from py2neo import Graph
import re

from tqdm import tqdm


class MedicalExtractor(object):
    def __init__(self, data_path):
        super(MedicalExtractor, self).__init__()
        self.data_path = data_path
        self.graph = Graph(
            profile="bolt://localhost:7687",
            auth=("neo4j", "general-judge-riviera-mozart-beyond-2404")
        )

        # 第一次清空数据库
        # self.graph.delete_all()
        # print('数据表已清空.')

        # 4 类节点(实体) (这里存的都是字符串)
        self.DRUG, self.entities_drugs = 'Drug', set() # 药品
        self.FOOD, self.entities_foods = 'Food', set() # 食物
        self.DISEASE, self.entities_diseases = 'Disease', set() # 疾病
        self.SYMPTOM, self.entities_symptoms = 'Symptom', set() # 症状

        # 4 类关系 (这里存的都是字符串)
        self.NOT_EAT, self.rels_not_eat = 'not_eat', set() # 忌吃关系 ((疾病，食物),)
        self.DO_EAT, self.rels_do_eat = 'do_eat', set() # 应吃关系 ((疾病，食物),)
        self.RECOMMAND_DRUG, self.rels_recommand_drug = 'recommand_drug', set() # 推荐药品关系 ((疾病，药品),)
        self.HAS_SYMPTOM, self.rels_symptom = 'has_symptom', set() # 症状关系 ((疾病，症状),)

        print('MedicalExtractor 创建完成.')

    def extract_entities_and_pairs(self):
        with open(self.data_path, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), desc='数据抽取中...'):
                data_json = json.loads(line)

                # 当前疾病名称
                disease = data_json['name']
                self.entities_diseases.add(disease)

                # 症状
                symptoms = data_json.get('symptom', [])
                self.entities_symptoms.update(symptoms)
                self.rels_symptom.update([(disease, symptom) for symptom in symptoms])

                # 并发症
                self.entities_diseases.update(data_json.get('acompany', []))

                # 推荐药品
                recommand_drugs = data_json.get('recommand_drug', [])
                self.entities_drugs.update(recommand_drugs)
                self.rels_recommand_drug.update([(disease, drug) for drug in recommand_drugs])

                # 忌吃
                not_eat_foods = data_json.get('not_eat', [])
                self.entities_foods.update(not_eat_foods)
                self.rels_not_eat.update([(disease, food) for food in not_eat_foods])

                # 建议吃
                do_eat_foods = data_json.get('do_eat', [])
                self.entities_foods.update(do_eat_foods)
                self.rels_do_eat.update([(disease, food) for food in do_eat_foods])

                # 疾病相关药品
                # ['桂林南药布美他尼片(布美他尼片)', ...] 这里保存括号里的简称
                detail_drugs = data_json.get('drug_detail', [])
                for drug in detail_drugs:
                    splits = re.split(r'[()]', drug)
                    self.entities_drugs.add(splits[1] if len(splits) == 3 else splits[0])

        print('\n抽取完成... 共有：')
        print(f'{self.DRUG}: {len(self.entities_drugs)}')
        print(f'{self.FOOD}: {len(self.entities_foods)}')
        print(f'{self.DISEASE}: {len(self.entities_diseases)}')
        print(f'{self.SYMPTOM}: {len(self.entities_symptoms)}')
        print('--------------')
        print(f'{self.NOT_EAT}: {len(self.rels_not_eat)}')
        print(f'{self.DO_EAT}: {len(self.rels_do_eat)}')
        print(f'{self.RECOMMAND_DRUG}: {len(self.rels_recommand_drug)}')
        print(f'{self.HAS_SYMPTOM}: {len(self.rels_symptom)}')
        print('-----------------------')

    def batch_run_cql(self, cql_list, batch_size=1000, desc=None):
        for i in tqdm(range(0, len(cql_list), batch_size), desc=f'{desc} (batch_size={batch_size})'):
            try:
                tx = self.graph.begin()  # 开启事务
                for cql in cql_list[i:i+batch_size]:
                    tx.run(cql)
                self.graph.commit(tx) # 提交事务
            except Exception as e:
                print(e)

    def write_nodes(self, entity_type, entity_list):
        get_cql = lambda entity: """
            MERGE (n:{entity_type}{{name: '{entity}'}}) RETURN n;
        """.format(
            entity_type=entity_type,
            entity=entity.replace("'", "")
        )
        # MERGE (n:Disease{name: '肺泡蛋白质沉积症'}) RETURN n;

        sql_list = [get_cql(entity) for entity in entity_list]
        self.batch_run_cql(sql_list, desc=f'{entity_type} 节点正在写入数据库...')

    def write_edges(self, pairs, head_entity_type, relation_type, tail_entity_type):
        get_cql = lambda head_name, tail_name:"""
            MATCH (a:{head_label} {{name: '{head_name}'}}),
                  (b:{tail_label} {{name: '{tail_name}'}})
            MERGE (a)-[r:{relation}]->(b)
            RETURN r;
        """.format(
            head_label=head_entity_type,
            tail_label=tail_entity_type,
            relation=relation_type,
            head_name=head_name.replace("'", ""),
            tail_name=tail_name.replace("'", "")
        )
        # MATCH (a:Disease {name: 癌症}),
        #       (b:Drug {name: 阿莫西林})
        # MERGE (a)-[r:recommand_drug]->(b)
        # RETURN r;

        cql_list = [get_cql(head, tail) for head, tail in pairs]
        self.batch_run_cql(cql_list, desc=f'{relation_type} 关系正在写入数据库...')

    def create_entities(self):
        self.write_nodes(self.DISEASE, self.entities_diseases)
        self.write_nodes(self.FOOD, self.entities_foods)
        self.write_nodes(self.DRUG, self.entities_drugs)
        self.write_nodes(self.SYMPTOM, self.entities_symptoms)

    def create_relations(self):
        self.write_edges(self.rels_recommand_drug, self.DISEASE, self.RECOMMAND_DRUG, self.DRUG)
        self.write_edges(self.rels_not_eat, self.DISEASE, self.NOT_EAT, self.FOOD)
        self.write_edges(self.rels_do_eat, self.DISEASE, self.DO_EAT, self.FOOD)
        self.write_edges(self.rels_symptom, self.DISEASE, self.HAS_SYMPTOM, self.SYMPTOM)


if __name__ == '__main__':
    extractor = MedicalExtractor('./medical.json')
    extractor.extract_entities_and_pairs()
    extractor.create_entities()
    extractor.create_relations()
    print('程序运行结束.')