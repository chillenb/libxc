
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ft97_a_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.086710549804283e+01, -2.086712947743720e+01, -2.086731232759304e+01, -2.086691815222271e+01, -2.086711750973215e+01, -2.086711750973215e+01, -3.461280657107637e+00, -3.461250184164102e+00, -3.460556056803931e+00, -3.462486870687908e+00, -3.461279728414524e+00, -3.461279728414524e+00, -6.987430289680424e-01, -6.989207729253170e-01, -7.069046166953017e-01, -7.108935563150387e-01, -6.988031320591181e-01, -6.988031320591181e-01, -2.262072083777940e-01, -2.260603895633654e-01, -8.106143958177652e-01, -2.081298041415741e-01, -2.261171660852923e-01, -2.261171660852923e-01, -6.863438335726733e-02, -6.951833719408612e-02, -1.298710749834543e-01, -6.101095993602875e-02, -6.884045085460083e-02, -6.884045085460083e-02, -5.015712051614336e+00, -5.015172478939018e+00, -5.015662985071685e+00, -5.015243320843516e+00, -5.015426324524200e+00, -5.015426324524200e+00, -2.128059091429783e+00, -2.136407892534794e+00, -2.131427441136459e+00, -2.137861835330229e+00, -2.129535825214582e+00, -2.129535825214582e+00, -5.718072210470529e-01, -5.943063105842979e-01, -5.447534462205579e-01, -5.461901664815916e-01, -5.893520967564864e-01, -5.893520967564864e-01, -1.905855060326984e-01, -2.614730911501055e-01, -1.852717157568824e-01, -1.807890328558886e+00, -1.944492827908841e-01, -1.944492827908841e-01, -5.984245411783363e-02, -6.159909788258886e-02, -4.709003339132588e-02, -1.478351529369351e-01, -5.593216064874788e-02, -5.593216064874788e-02, -5.582708506947187e-01, -5.604764511459733e-01, -5.596836979505416e-01, -5.590701302558143e-01, -5.593762274903353e-01, -5.593762274903353e-01, -5.395257268666140e-01, -5.158932162302307e-01, -5.214884238479207e-01, -5.274195743604679e-01, -5.241154000538484e-01, -5.241154000538484e-01, -6.240312848099859e-01, -2.983541426696407e-01, -3.237222616021669e-01, -3.669512412636220e-01, -3.422764788651396e-01, -3.422764788651396e-01, -4.685955265975298e-01, -1.294065230427865e-01, -1.454071498987643e-01, -3.356400947829645e-01, -1.630821274308772e-01, -1.630821274308772e-01, -7.559814422784301e-02, -4.318147119323849e-02, -4.980074458525379e-02, -1.551303071570421e-01, -5.011599210865775e-02, -5.011599210865773e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ft97_a_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.506993224442971e+01, -2.507002647963117e+01, -2.507044342607862e+01, -2.506890034961427e+01, -2.506998196978612e+01, -2.506998196978612e+01, -4.002845778800089e+00, -4.002899689565495e+00, -4.004545512971003e+00, -4.002557237538829e+00, -4.002890558517327e+00, -4.002890558517327e+00, -7.273424236525933e-01, -7.256516926052229e-01, -6.799867829578564e-01, -6.868074077602945e-01, -7.267303387199736e-01, -7.267303387199736e-01, -1.724564409840773e-01, -1.762736267000463e-01, -9.013891365420467e-01, -1.244189268305765e-01, -1.736058156072459e-01, -1.736058156072459e-01, -2.637383637383785e-02, -2.716362210185413e-02, -7.342158619574501e-02, -1.768084817541695e-02, -2.688965741877628e-02, -2.688965741877628e-02, -6.195597287283968e+00, -6.198533687620766e+00, -6.195896988691643e+00, -6.198180242657781e+00, -6.197104368620611e+00, -6.197104368620611e+00, -2.054629804199314e+00, -2.075342834180344e+00, -2.040048803173616e+00, -2.056205935972642e+00, -2.087639414708073e+00, -2.087639414708073e+00, -6.751261567690863e-01, -7.676538560145393e-01, -6.379715045398692e-01, -6.932809921943336e-01, -7.065546368121915e-01, -7.065546368121915e-01, -1.119405602989492e-01, -1.671659293465452e-01, -1.086598703184391e-01, -2.334658984143497e+00, -1.136835263560047e-01, -1.136835263560047e-01, -1.722366254689506e-02, -1.852409205768818e-02, -1.404236449633185e-02, -8.740676889009756e-02, -1.684485260894417e-02, -1.684485260894416e-02, -7.375626098979717e-01, -7.267412385248705e-01, -7.306172684689117e-01, -7.336150933823425e-01, -7.321155071644452e-01, -7.321155071644452e-01, -7.149945054459947e-01, -5.608276095191654e-01, -6.061145129263442e-01, -6.483801791939107e-01, -6.270796300090256e-01, -6.270796300090256e-01, -8.036953578076779e-01, -2.139915324989604e-01, -2.685705088575635e-01, -3.834672588449661e-01, -3.218678815545275e-01, -3.218678815545274e-01, -5.021719204718635e-01, -7.124694799272928e-02, -8.479997837528164e-02, -3.746208610252375e-01, -9.627635644598319e-02, -9.627635644598309e-02, -2.996376924972682e-02, -9.255204669273041e-03, -1.258217122188132e-02, -9.155797595141482e-02, -1.475888767295448e-02, -1.475888767295438e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ft97_a_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.970326157991252e-09, -6.970264001828472e-09, -6.969928896604976e-09, -6.970947901232755e-09, -6.970293862216874e-09, -6.970293862216874e-09, -9.797999420430472e-06, -9.797994095474733e-06, -9.796230591254859e-06, -9.790960637145559e-06, -9.797829731048894e-06, -9.797829731048894e-06, -6.949231038431804e-03, -6.969367217413210e-03, -7.465164061011614e-03, -7.252123915486504e-03, -6.956615594457027e-03, -6.956615594457027e-03, -1.005444143099498e+00, -9.738847957501451e-01, -3.465936210246404e-03, -2.301537439960761e+00, -9.962024067044278e-01, -9.962024067044278e-01, -2.908295760900618e+03, -2.550631357423604e+03, -3.974319550771581e+01, -2.156926543622033e+04, -2.656863246183609e+03, -2.656863246183609e+03, -1.993128004795026e-06, -1.992012704209640e-06, -1.993008683251857e-06, -1.992141562631683e-06, -1.992563594586357e-06, -1.992563594586357e-06, -9.039912080741617e-05, -8.819148113033945e-05, -9.099328480359694e-05, -8.925694990935940e-05, -8.813279248632059e-05, -8.813279248632059e-05, -1.272091227070307e-02, -9.349998076024109e-03, -1.564813874904083e-02, -1.352202994684553e-02, -1.099320613908953e-02, -1.099320613908953e-02, -4.545837146790321e+00, -7.662596765465491e-01, -4.985465322318496e+00, -1.092327209633370e-04, -3.517269042087913e+00, -3.517269042087913e+00, -2.431691007195759e+04, -1.658499740910351e+04, -5.106510537418014e+04, -1.702391855684432e+01, -2.418473320544749e+04, -2.418473320544750e+04, -1.151356772990579e-02, -1.173778039038861e-02, -1.165884939212531e-02, -1.159699419579728e-02, -1.162807873798641e-02, -1.162807873798641e-02, -1.312042367998607e-02, -2.188398781529071e-02, -1.886265181586764e-02, -1.643490216097675e-02, -1.763440737793167e-02, -1.763440737793166e-02, -7.732604753930519e-03, -3.659680678386717e-01, -2.110511745389547e-01, -9.082321446772719e-02, -1.404439609153388e-01, -1.404439609153389e-01, -3.286607579226775e-02, -4.605674448178721e+01, -2.088339797617724e+01, -1.172313259471480e-01, -9.101818201255535e+00, -9.101818201255540e+00, -1.704660476596188e+03, -8.041592668174957e+05, -1.212785107769424e+05, -1.107307557071669e+01, -4.293013459217233e+04, -4.293013459217253e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05