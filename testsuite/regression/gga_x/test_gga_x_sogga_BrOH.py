
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sogga_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.047138931797287e+01, -2.047142296969010e+01, -2.047163288656735e+01, -2.047108063802521e+01, -2.047140656496582e+01, -2.047140656496582e+01, -3.370929296617384e+00, -3.370914153819441e+00, -3.370637170462353e+00, -3.371829477213909e+00, -3.370935867911104e+00, -3.370935867911104e+00, -6.663654819571310e-01, -6.661808259282560e-01, -6.630040355556497e-01, -6.673747751884000e-01, -6.662955677680633e-01, -6.662955677680633e-01, -2.004169128534316e-01, -2.015573369027451e-01, -7.834175618279196e-01, -1.639762059671892e-01, -2.007387400885777e-01, -2.007387400885777e-01, -1.465052174132962e-02, -1.534387045160324e-02, -6.155894469552315e-02, -7.035340760765426e-03, -1.518401102168815e-02, -1.518401102168815e-02, -4.946361342536171e+00, -4.946365715608843e+00, -4.946366625530636e+00, -4.946369942054625e+00, -4.946356584613663e+00, -4.946356584613663e+00, -1.997506381044316e+00, -2.007993890511063e+00, -1.996888983950552e+00, -2.005042348188455e+00, -2.005464139769727e+00, -2.005464139769727e+00, -5.591078125263300e-01, -5.909298440107481e-01, -5.318240693228309e-01, -5.413572110236106e-01, -5.779435720305670e-01, -5.779435720305670e-01, -1.308692137133693e-01, -2.169885696372219e-01, -1.286313248724230e-01, -1.797541120879742e+00, -1.450923902517021e-01, -1.450923902517021e-01, -6.786969175605046e-03, -7.755849650950413e-03, -5.813135340152602e-03, -8.464894470857465e-02, -7.066762525709516e-03, -7.066762525709516e-03, -5.573596272633775e-01, -5.576802009367858e-01, -5.575693130923747e-01, -5.574784706971913e-01, -5.575237549832714e-01, -5.575237549832714e-01, -5.389410177959321e-01, -4.963426063446782e-01, -5.083684268906089e-01, -5.196625168811064e-01, -5.137730815650257e-01, -5.137730815650255e-01, -6.201563403264028e-01, -2.594530616251509e-01, -2.933650750646740e-01, -3.502291930244306e-01, -3.194311172152172e-01, -3.194311172152172e-01, -4.495405903389174e-01, -5.718655855175241e-02, -7.669427101735529e-02, -3.246183405497707e-01, -1.076232479003509e-01, -1.076232479003509e-01, -1.716625550299305e-02, -2.123992382553035e-03, -4.034532215558849e-03, -1.026079575735646e-01, -6.006044193003225e-03, -6.006044193003216e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sogga_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.563975605846657e+01, -2.563983588327530e+01, -2.564021239960843e+01, -2.563890483231960e+01, -2.563979798483454e+01, -2.563979798483454e+01, -4.134150738663347e+00, -4.134181888284087e+00, -4.135208215279949e+00, -4.134316040263736e+00, -4.134184370818243e+00, -4.134184370818243e+00, -7.744192277187821e-01, -7.732388676714514e-01, -7.427411902816010e-01, -7.490947957365428e-01, -7.739903596564806e-01, -7.739903596564806e-01, -2.051216403424179e-01, -2.078332483093089e-01, -9.410666861290261e-01, -1.632996640002190e-01, -2.059226920785000e-01, -2.059226920785000e-01, -1.948604602912559e-02, -2.040332160449713e-02, -7.988764620659188e-02, -9.375965993888971e-03, -2.019085346619784e-02, -2.019085346619784e-02, -6.294364864421548e+00, -6.296499292497037e+00, -6.294584421878392e+00, -6.296244047762354e+00, -6.295457673766572e+00, -6.295457673766572e+00, -2.241452133631157e+00, -2.259421497042768e+00, -2.232079397840835e+00, -2.246060143687929e+00, -2.265927280393512e+00, -2.265927280393512e+00, -6.935037485827302e-01, -7.723248858589544e-01, -6.567168282294404e-01, -7.000574230351384e-01, -7.229938136588450e-01, -7.229938136588450e-01, -1.476510203090307e-01, -2.133689075719221e-01, -1.435954730490942e-01, -2.348979117116016e+00, -1.507509323591926e-01, -1.507509323591926e-01, -9.045141719102225e-03, -1.033493421777304e-02, -7.746431272948597e-03, -1.066840489369437e-01, -9.416651492220991e-03, -9.416651492220991e-03, -7.387938133453640e-01, -7.305963708635890e-01, -7.335134047796895e-01, -7.357832009615255e-01, -7.346460318533494e-01, -7.346460318533494e-01, -7.157813019834637e-01, -5.893532895147860e-01, -6.251647709701527e-01, -6.594520541491606e-01, -6.419917490599245e-01, -6.419917490599244e-01, -8.090693098395048e-01, -2.607319367311925e-01, -3.092634844150154e-01, -4.077931560491924e-01, -3.542913143495683e-01, -3.542913143495683e-01, -5.299566769550477e-01, -7.461537646992671e-02, -9.834793980243134e-02, -3.906979837316856e-01, -1.257547840348335e-01, -1.257547840348335e-01, -2.282149451761312e-02, -2.831822708487558e-03, -5.378313648026953e-03, -1.196732787279404e-01, -8.003830600955273e-03, -8.003830600955261e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sogga_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.191575923631223e-09, -4.191544839808599e-09, -4.191362300240882e-09, -4.191872190144435e-09, -4.191559897784389e-09, -4.191559897784389e-09, -5.768440812448363e-06, -5.768506827415965e-06, -5.769368487768172e-06, -5.762992752311925e-06, -5.768376475175390e-06, -5.768376475175390e-06, -3.879723238215911e-03, -3.886163767722628e-03, -4.016566326488624e-03, -3.909781771385379e-03, -3.882121656384305e-03, -3.882121656384305e-03, -4.834491966407086e-01, -4.740217898008610e-01, -1.999051668934301e-03, -8.319178911341001e-01, -4.808628546575881e-01, -4.808628546575881e-01, -2.142331535951115e+00, -2.147560794970289e+00, -8.730234343530240e-01, -1.520573371023790e+00, -2.230969675811067e+00, -2.230969675811067e+00, -1.218449830943895e-06, -1.218201632625058e-06, -1.218420244022255e-06, -1.218227328790445e-06, -1.218328632874541e-06, -1.218328632874541e-06, -4.872166603230354e-05, -4.766618815005650e-05, -4.884562397816147e-05, -4.802123816826425e-05, -4.783607861138077e-05, -4.783607861138077e-05, -7.574724314075102e-03, -5.885286876526383e-03, -9.275918242882302e-03, -8.410415505266734e-03, -6.602682116446411e-03, -6.602682116446411e-03, -8.582194482345589e-01, -3.207056094084998e-01, -1.005709444951431e+00, -6.874413179674803e-05, -1.031768025136491e+00, -1.031768025136491e+00, -1.613132616055750e+00, -1.616220203589847e+00, -4.626089396801735e+00, -9.591789884896832e-01, -2.387224474659181e+00, -2.387224474659179e+00, -7.370886214369574e-03, -7.408498129030896e-03, -7.394999218703177e-03, -7.384677417467585e-03, -7.389861155270185e-03, -7.389861155270185e-03, -8.420827935917993e-03, -1.247899741299419e-02, -1.113526613907647e-02, -1.001788597755900e-02, -1.057928863528799e-02, -1.057928863528798e-02, -4.857248642276664e-03, -1.696415177889778e-01, -1.059920124261255e-01, -5.080134420795494e-02, -7.478138546830121e-02, -7.478138546830124e-02, -1.860934443819192e-02, -7.426582103672198e-01, -7.488049248267291e-01, -6.774474270704571e-02, -1.332814706088965e+00, -1.332814706088967e+00, -1.608696205293402e+00, -2.781515746208439e+00, -2.394614344108255e+00, -1.646070085705980e+00, -3.486630564066678e+00, -3.486630564066671e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05