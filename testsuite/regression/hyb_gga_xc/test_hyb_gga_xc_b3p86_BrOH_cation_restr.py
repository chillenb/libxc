
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3p86_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.688061169687841e+01, -1.688063486014158e+01, -1.688079032704121e+01, -1.688044462135028e+01, -1.688061830007040e+01, -1.688061830007040e+01, -2.840118370961436e+00, -2.840093699313180e+00, -2.839592033257951e+00, -2.840994703127772e+00, -2.840156078235643e+00, -2.840156078235643e+00, -5.955336113330311e-01, -5.952720288312980e-01, -5.901970830673567e-01, -5.943250256745555e-01, -5.930337780796857e-01, -5.930337780796857e-01, -1.858268426780803e-01, -1.873838222376972e-01, -6.846977478261014e-01, -1.564117470043270e-01, -1.668067066387854e-01, -1.668067066387854e-01, -5.187906804828345e-02, -5.208744829202933e-02, -9.685676372303209e-02, -4.617750983470657e-02, -4.656248228296232e-02, -4.656248228296229e-02, -4.111606577739257e+00, -4.111295899933021e+00, -4.111600034881818e+00, -4.111325637157838e+00, -4.111444695528267e+00, -4.111444695528267e+00, -1.722454783513029e+00, -1.730931797474536e+00, -1.722276438952173e+00, -1.729736678300814e+00, -1.727255141373598e+00, -1.727255141373598e+00, -5.137150031073205e-01, -5.451923342946902e-01, -4.797794533667697e-01, -4.887267060689833e-01, -5.200612170385409e-01, -5.200612170385409e-01, -1.314446803900069e-01, -1.953110462906751e-01, -1.261557389381875e-01, -1.534670820706729e+00, -1.369064687613304e-01, -1.369064687613304e-01, -4.138312337120668e-02, -4.453408886032086e-02, -2.957133521468185e-02, -1.061857756058436e-01, -3.567978876846904e-02, -3.567978876846906e-02, -5.058179328687640e-01, -5.047873887480707e-01, -5.051236416111629e-01, -5.054172781595698e-01, -5.052675057101662e-01, -5.052675057101662e-01, -4.928910682528465e-01, -4.457149059399438e-01, -4.585000768594207e-01, -4.715144151475162e-01, -4.647436338994188e-01, -4.647436338994188e-01, -5.697233736623138e-01, -2.327668733605331e-01, -2.656593709387103e-01, -3.258067530622249e-01, -2.932194705342180e-01, -2.932194705342179e-01, -4.134163281482107e-01, -9.722772089474443e-02, -1.050085953035527e-01, -3.117554309626434e-01, -1.115505869891827e-01, -1.115505869891826e-01, -5.929367042793195e-02, -2.501397699050042e-02, -3.326956215266499e-02, -1.092351767469806e-01, -3.352373167231457e-02, -3.352373167231455e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3p86_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.035261664445525e+01, -2.035269131336834e+01, -2.035301169550957e+01, -2.035190102307544e+01, -2.035248463283780e+01, -2.035248463283780e+01, -3.369663837603572e+00, -3.369702968945014e+00, -3.370687687621150e+00, -3.369593045604489e+00, -3.369789048334937e+00, -3.369789048334937e+00, -7.055477808522651e-01, -7.044320837904787e-01, -6.765939167581319e-01, -6.821654455036860e-01, -6.814267717135228e-01, -6.814267717135228e-01, -2.027150516722223e-01, -2.063460229161595e-01, -8.151978415390329e-01, -1.482196660328333e-01, -1.709028580281685e-01, -1.709028580281685e-01, -1.944355626497456e-02, -1.995384841221201e-02, -4.230942529206869e-02, -1.426269017766164e-02, -1.588737375471589e-02, -1.588737375471592e-02, -5.109278089011475e+00, -5.111505477384090e+00, -5.109377809356540e+00, -5.111344178177457e+00, -5.110408442071297e+00, -5.110408442071297e+00, -1.848361575352516e+00, -1.863947345246662e+00, -1.832608535354737e+00, -1.846328619502952e+00, -1.864672577077711e+00, -1.864672577077711e+00, -6.399231651967132e-01, -7.035768941583448e-01, -5.943855870931239e-01, -6.273404644329187e-01, -6.510364624197361e-01, -6.510364624197361e-01, -9.051810485082255e-02, -1.928603171256176e-01, -8.208333178946570e-02, -1.987505111284707e+00, -1.175098446485084e-01, -1.175098446485084e-01, -1.202441901678298e-02, -1.386003163356437e-02, -9.961875111223336e-03, -5.137034649862909e-02, -1.194105074608570e-02, -1.194105074608573e-02, -6.575881933726568e-01, -6.514803984196251e-01, -6.536503860744213e-01, -6.554282827652537e-01, -6.545408234813789e-01, -6.545408234813789e-01, -6.417965834340841e-01, -5.356119716878778e-01, -5.661188143749711e-01, -5.965642564506973e-01, -5.811018042495414e-01, -5.811018042495414e-01, -7.344692626895486e-01, -2.439583038480680e-01, -2.945669179648281e-01, -3.919500972216924e-01, -3.402819801478488e-01, -3.402819801478487e-01, -4.972723198711800e-01, -4.317296390180924e-02, -4.418549368268366e-02, -3.834175391975813e-01, -7.133034208560562e-02, -7.133034208560564e-02, -2.434616123698336e-02, -5.839049133895234e-03, -9.243055154607004e-03, -6.534499399011355e-02, -1.116667891535629e-02, -1.116667891535625e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3p86_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.391663712854368e-09, -5.391637798543488e-09, -5.391448851113436e-09, -5.391835926534749e-09, -5.391643587847860e-09, -5.391643587847860e-09, -6.416139476511762e-06, -6.416197693291324e-06, -6.416933118608727e-06, -6.410992524557373e-06, -6.415618857275871e-06, -6.415618857275871e-06, -2.743542168390935e-03, -2.754034586241791e-03, -3.007560916251535e-03, -2.921706335335732e-03, -2.941079523132651e-03, -2.941079523132651e-03, -2.948091733245907e-01, -2.802495522489349e-01, -1.586181825867506e-03, -7.690539417452438e-01, -5.120251105012347e-01, -5.120251105012349e-01, -7.302090761948234e+03, -6.405337733752288e+03, -5.098893245674672e+01, -3.444042436886041e+04, -1.908917584871586e+04, -1.908917584871587e+04, -1.418811035876517e-06, -1.417909781272305e-06, -1.418765675444223e-06, -1.417970332278917e-06, -1.418358294863493e-06, -1.418358294863493e-06, -5.096629747138289e-05, -4.980119022535005e-05, -5.145992161883330e-05, -5.041970455264839e-05, -5.008030010364242e-05, -5.008030010364242e-05, -4.331973958683149e-03, -2.830166713260378e-03, -5.735395052697901e-03, -4.569240699463506e-03, -4.066660262733288e-03, -4.066660262733288e-03, -2.759611022222481e+00, -2.896032746665643e-01, -3.628226211817977e+00, -5.240096304466958e-05, -1.600502047050152e+00, -1.600502047050152e+00, -7.855502302065115e+04, -3.847918142202037e+04, -1.396238430301055e+05, -1.386823480039808e+01, -6.585838820380703e+04, -6.585838820380699e+04, -3.187908598517668e-03, -3.807421470042340e-03, -3.646344700193148e-03, -3.477366121959737e-03, -3.567050971817360e-03, -3.567050971817360e-03, -3.109759817502100e-03, -8.179021696396805e-03, -6.899816044252294e-03, -5.721144509042683e-03, -6.318249458091097e-03, -6.318249458091097e-03, -2.417397917177778e-03, -1.298621071008421e-01, -7.015408883557211e-02, -2.755391578695885e-02, -4.419335023464906e-02, -4.419335023464907e-02, -1.092920211790443e-02, -5.512940105354498e+01, -2.455134502513970e+01, -3.140711287900767e-02, -6.478059675382223e+00, -6.478059675382225e+00, -2.563280759984066e+03, -2.703133770297323e+06, -2.592679251634672e+05, -8.007711189354167e+00, -8.792920269043498e+04, -8.792920269043527e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05