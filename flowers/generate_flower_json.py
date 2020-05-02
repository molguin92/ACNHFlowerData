from __future__ import annotations

import itertools
import json
from enum import IntEnum
from typing import Tuple

import numpy as np
import pandas as pd

_flower_data = pd.read_csv('../data/flower_data.csv',
                           dtype={'species': 'category',
                                  'r'      : 'int8',
                                  'y'      : 'int8',
                                  'w'      : 'int8',
                                  's'      : 'int8',
                                  'color'  : 'category',
                                  'origin' : 'category'},
                           index_col=['species', 'r', 'y', 'w', 's'])


class Allele(IntEnum):
    rec = 0
    dom = 1


class Gene(IntEnum):
    gg = 0
    Gg = 1
    GG = 2

    @classmethod
    def fromAlleles(cls, a1: Allele, a2: Allele) -> Gene:
        return Gene(a1.value + a2.value)

    def getAlleles(self) -> Tuple[Allele, Allele]:
        if self.value == 0:
            return Allele(0), Allele(0)
        elif self.value == 1:
            return Allele(1), Allele(0)
        else:
            return Allele(1), Allele(1)

    def getAllOffspring(self, other: Gene) -> Tuple[Gene, ...]:
        alleles_l = self.getAlleles()
        alleles_r = other.getAlleles()

        return tuple(Gene.fromAlleles(a1, a2)
                     for a1, a2 in itertools.product(alleles_l, alleles_r))


class Flower:
    def __init__(self,
                 gR: Gene,
                 gY: Gene,
                 gW: Gene,
                 gS: Gene):
        self.red = gR
        self.ylw = gY
        self.wht = gW
        self.shd = gS

    def __gt__(self, other):
        return self.__hash__() > other.__hash__()

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return (self.red * 1000) + \
               (self.ylw * 100) + \
               (self.wht * 10) + \
               self.shd

    def __str__(self):
        return f'{self.red.value}' \
               f'{self.ylw.value}' \
               f'{self.wht.value}' \
               f'{self.shd.value}'

    def __repr__(self):
        return self.__hash__()

    def getAllOffspring(self, other: Flower) -> Tuple[Flower, ...]:
        red = self.red.getAllOffspring(other.red)
        ylw = self.ylw.getAllOffspring(other.ylw)
        wht = self.wht.getAllOffspring(other.wht)
        shd = self.shd.getAllOffspring(other.shd)

        return tuple(Flower(r, y, w, s)
                     for r, y, w, s in itertools.product(red, ylw, wht, shd))


def order(x, y):
    return max(x, y), min(x, y)


if __name__ == '__main__':
    data = _flower_data.copy()

    flower_db = {}

    for species, df in data.groupby(level=0):
        df = df.copy().reset_index()
        flowers = []
        variants = {}
        for row in df.itertuples(index=False):
            f = Flower(Gene(row.r), Gene(row.y),
                       Gene(row.w), Gene(row.s))
            flowers.append(f)
            variants[f.__hash__()] = {'color': row.color, 'origin': row.origin}

        matings = {}
        for mate1, mate2 in itertools.product(flowers, flowers):
            mate1, mate2 = order(mate1, mate2)
            if mate1.__hash__() in matings \
                    and mate2.__hash__() in matings[mate1.__hash__()]:
                continue

            offspring = mate1.getAllOffspring(mate2)
            values, counts = np.unique([x.__hash__() for x in offspring],
                                       return_counts=True)
            for v in values:
                assert v in variants
            count_sum = np.sum(counts)
            offspring = {int(v): c / count_sum
                         for v, c in zip(list(values), list(counts))}

            if mate1.__hash__() not in matings:
                matings[mate1.__hash__()] = {}

            matings[mate1.__hash__()][mate2.__hash__()] = offspring

        flower_db[species] = {'variants': variants, 'matings': matings}

    with open('../data/flower_data.json', 'w') as fp:
        json.dump(flower_db, fp, indent=4)
