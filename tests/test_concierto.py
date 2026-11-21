import unittest
import numpy as np
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ticket_pricing.concierto import Evento, Compradores, estudio, ejecutar, compra, rejilla, Plan, TUNE, SELECT, FINAL

class ProyectoTests(unittest.TestCase):
    def test_limites(self):
        with self.assertRaises(ValueError):
            Evento(minimo=41)
        with self.assertRaises(ValueError):
            Evento(referencia=250)
    def test_llegadas_y_cuantiles(self):
        c=Evento()
        self.assertAlmostEqual(c.llegadas.sum(),c.demanda*c.capacidad)
        self.assertTrue(np.all(np.diff(c.llegadas)>0))
        self.assertLess(compra(120,100,c),compra(120,120,c))
        self.assertGreater(compra(80,100,c),compra(80,80,c))
    def test_compradores_comunes(self):
        c=Evento(capacidad=100,dias=9)
        tape=Compradores(c,(120,20));same=Compradores(c,(120,20))
        a=tape.interesados(2,.6,False)
        np.testing.assert_array_equal(a,same.interesados(2,.6,False))
        self.assertTrue(np.all(a<=tape.interesados(2,.7,False)))
        self.assertTrue(np.all(a<=tape.interesados(2,.6,True)))
    def test_pfa_reproduce_fijo_e_inventario(self):
        c=Evento(capacidad=100,dias=10);t=Compradores(c,(9,40))
        a=ejecutar(c,t,"fixed",100,trace=True); b=ejecutar(c,t,"rule",[100,0,0,0],trace=True)
        np.testing.assert_equal(a["ingresos"],b["ingresos"])
        np.testing.assert_equal(a["ingresos"],(100-a["restantes"])*100)
        self.assertTrue(np.all(np.diff(a["stock"],axis=0)<=0))
    def test_ultimo_dia_y_rejilla(self):
        c=Evento(dias=10,capacidad=100);p=Plan(c);r=95;stock=55
        best=c.precios[np.argmax(c.precios*np.minimum(stock,c.llegadas[-1]*compra(c.precios,r,c)))]
        self.assertEqual(p.precio(c.dias-1,stock,r),best)
        self.assertTrue(np.isin(rejilla([-10,107,900],c),c.precios).all())
    def test_final_completo(self):
        self.assertEqual(len(set(x[0] for x in (TUNE,SELECT,FINAL))),3)
        c=Evento(dias=14,capacidad=400); s=estudio(c)
        for r,x in zip(s["resultados"],s["stats"]):
            self.assertEqual(len(r["ingresos"]),600)
            self.assertTrue(np.all(r["restantes"]>=0))
            self.assertTrue(np.isin(r["precios"][np.isfinite(r["precios"])],c.precios).all())
            self.assertTrue(0<=x["lleno"]<=1)
            self.assertAlmostEqual(x["vendidos"],c.capacidad-r["restantes"].mean())
            self.assertLessEqual(x["ic"][0],x["media"])
    def test_memoria_nula(self):
        c=replace(Evento(),eta=0,gamma=0)
        self.assertEqual(compra(100,40,c),compra(100,200,c))

if __name__=="__main__": unittest.main()
