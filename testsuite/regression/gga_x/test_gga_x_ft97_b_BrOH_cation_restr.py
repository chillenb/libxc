
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ft97_b_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.119174913162493e+01, -2.119176845426284e+01, -2.119193641279135e+01, -2.119164725036097e+01, -2.119178709174886e+01, -2.119178709174886e+01, -3.465464316691548e+00, -3.465423223031559e+00, -3.464559773322712e+00, -3.466727467936392e+00, -3.465499538908064e+00, -3.465499538908064e+00, -6.968661535973794e-01, -6.969816201470612e-01, -7.027362172364415e-01, -7.073033579328746e-01, -7.053241653480927e-01, -7.053241653480927e-01, -2.301294409660066e-01, -2.298052042532651e-01, -8.034648116466550e-01, -2.143571752169617e-01, -2.182450758064605e-01, -2.182450758064605e-01, -6.906476296275080e-02, -6.913308007610131e-02, -1.274712509800144e-01, -6.288691270534742e-02, -6.268219258697298e-02, -6.268219258697294e-02, -5.046977210191411e+00, -5.046102452362806e+00, -5.046949090454604e+00, -5.046176633699061e+00, -5.046526811350067e+00, -5.046526811350067e+00, -2.126423427101825e+00, -2.135112254210008e+00, -2.130715594872858e+00, -2.138300366513217e+00, -2.129234161336241e+00, -2.129234161336241e+00, -5.783943974707462e-01, -6.015423309735215e-01, -5.395464033211911e-01, -5.358917036082367e-01, -5.841044487274659e-01, -5.841044487274659e-01, -1.923639716985344e-01, -2.610470293204461e-01, -1.845833170434158e-01, -1.809945455357968e+00, -1.940765709930888e-01, -1.940765709930888e-01, -5.668642169627875e-02, -6.058319877107447e-02, -3.971851552020732e-02, -1.504491946539732e-01, -4.804086606721093e-02, -4.804086606721095e-02, -5.503064278004255e-01, -5.527801476167458e-01, -5.519077283080102e-01, -5.511832950323168e-01, -5.515446276055894e-01, -5.515446276055894e-01, -5.337666757771986e-01, -5.100023800909494e-01, -5.151658081626391e-01, -5.212260947648648e-01, -5.177819207535539e-01, -5.177819207535539e-01, -6.316660101206730e-01, -2.970052670953828e-01, -3.221428180634163e-01, -3.661817801505031e-01, -3.403733863511488e-01, -3.403733863511489e-01, -4.708169330794416e-01, -1.272684627165187e-01, -1.430612569595297e-01, -3.412849529094683e-01, -1.616072311350587e-01, -1.616072311350587e-01, -7.805225199281327e-02, -3.488621529644724e-02, -4.573124302471936e-02, -1.578385176393489e-01, -4.514277589967441e-02, -4.514277589967438e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ft97_b_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.468181404295817e+01, -2.468193034579564e+01, -2.468239249541865e+01, -2.468066326811541e+01, -2.468157715729026e+01, -2.468157715729026e+01, -3.998313160403713e+00, -3.998380125666137e+00, -4.000024692736658e+00, -3.997915420370123e+00, -3.998488717176648e+00, -3.998488717176648e+00, -7.385776069856689e-01, -7.368057347365216e-01, -6.923431073815367e-01, -6.993207093120992e-01, -6.988202174756520e-01, -6.988202174756520e-01, -1.587101042897936e-01, -1.619964260035295e-01, -8.712681913063642e-01, -1.263488584878426e-01, -1.350610422485727e-01, -1.350610422485727e-01, -2.075353763025263e-02, -2.122791262294902e-02, -6.658716052651073e-02, -1.579738436828190e-02, -1.718413288085404e-02, -1.718413288085406e-02, -6.161653343925023e+00, -6.165060013339552e+00, -6.161803471618210e+00, -6.164811037810099e+00, -6.163384133293104e+00, -6.163384133293104e+00, -2.060529217941907e+00, -2.081849035005708e+00, -2.035178262584017e+00, -2.053954099283856e+00, -2.084665180920559e+00, -2.084665180920559e+00, -6.822312882995073e-01, -7.753794830200984e-01, -6.259523067624688e-01, -6.822124680596063e-01, -6.977056892430815e-01, -6.977056892430815e-01, -1.134832043906097e-01, -1.595573343603613e-01, -1.091541623352597e-01, -2.337473242986711e+00, -1.130856365517816e-01, -1.130856365517816e-01, -1.348516073995465e-02, -1.529960094756763e-02, -1.062427116542354e-02, -8.778788842693835e-02, -1.283059809803432e-02, -1.283059809803431e-02, -7.253061828149285e-01, -7.136823178440831e-01, -7.177674429416037e-01, -7.211466651208833e-01, -7.194562028946464e-01, -7.194562028946464e-01, -7.078070900938982e-01, -5.438856145694230e-01, -5.911905860144766e-01, -6.385488086523757e-01, -6.146351963107907e-01, -6.146351963107907e-01, -8.115296886992377e-01, -1.998641718706244e-01, -2.508945830239780e-01, -3.774673462542841e-01, -3.078455493632998e-01, -3.078455493632999e-01, -4.993184756675260e-01, -6.519753851604002e-02, -7.955188447981244e-02, -3.767073087691704e-01, -9.564642757977128e-02, -9.564642757977126e-02, -2.587845550851785e-02, -6.929453285217300e-03, -1.043289000032790e-02, -9.342392416772678e-02, -1.198565003113546e-02, -1.198565003113547e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ft97_b_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.048444059829167e-09, -9.048357221754422e-09, -9.047917400711113e-09, -9.049210503606219e-09, -9.048540568545443e-09, -9.048540568545443e-09, -1.017070454232030e-05, -1.017064909667443e-05, -1.016844050630778e-05, -1.016512785150909e-05, -1.016973042757972e-05, -1.016973042757972e-05, -6.797458023995883e-03, -6.819647958913590e-03, -7.337912713481822e-03, -7.112499574545982e-03, -7.170331177896891e-03, -7.170331177896891e-03, -1.098495673532510e+00, -1.064264713526933e+00, -3.713698036610075e-03, -2.163368143173296e+00, -1.693410231901872e+00, -1.693410231901872e+00, -1.054059368336244e+04, -9.243110027951217e+03, -6.198504002874028e+01, -4.971708449023185e+04, -2.756409636566125e+04, -2.756409636566125e+04, -2.455906421231293e-06, -2.453627913982888e-06, -2.455799124436383e-06, -2.453787890725740e-06, -2.454754460832169e-06, -2.454754460832169e-06, -8.981799057555052e-05, -8.755721669916878e-05, -9.101695671305321e-05, -8.899161421503830e-05, -8.798706275344327e-05, -8.798706275344327e-05, -1.210362297332799e-02, -8.892050127584167e-03, -1.641929451720824e-02, -1.443533997939517e-02, -1.139924839961371e-02, -1.139924839961371e-02, -4.817367593916003e+00, -8.557284565578175e-01, -5.971291869200257e+00, -1.081278246523203e-04, -3.781063997138786e+00, -3.781063997138786e+00, -1.133580140757165e+05, -5.554865324911402e+04, -2.013509729434059e+05, -1.793974233182558e+01, -9.506141178071639e+04, -9.506141178071634e+04, -1.218232063525174e-02, -1.243315820135466e-02, -1.234649430424121e-02, -1.227398555374405e-02, -1.231043705453028e-02, -1.231043705453028e-02, -1.360271118897314e-02, -2.347147386025476e-02, -2.010348272372954e-02, -1.723520325264152e-02, -1.865120129374835e-02, -1.865120129374835e-02, -7.356928232217869e-03, -4.136047610109532e-01, -2.353604946525154e-01, -9.300082190911502e-02, -1.512811958234481e-01, -1.512811958234482e-01, -3.259188711599947e-02, -6.794906181490457e+01, -2.924654905316255e+01, -1.109812720399199e-01, -1.047547231861061e+01, -1.047547231861061e+01, -3.693785344176953e+03, -3.893597459362717e+06, -3.740032413785734e+05, -1.216412540893607e+01, -1.269001533498381e+05, -1.269001533498386e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05