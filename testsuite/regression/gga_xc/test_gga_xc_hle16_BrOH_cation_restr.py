
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hle16_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.638072303824611e+01, -2.638080096594397e+01, -2.638116286032021e+01, -2.638000315730135e+01, -2.638060861010583e+01, -2.638060861010583e+01, -4.326842216016977e+00, -4.326847981492648e+00, -4.327087966072406e+00, -4.327487710111450e+00, -4.326952598281157e+00, -4.326952598281157e+00, -8.636400296832080e-01, -8.631257077112707e-01, -8.545620263798194e-01, -8.606501990751184e-01, -8.587022062704633e-01, -8.587022062704633e-01, -2.815582075187349e-01, -2.811866592248259e-01, -9.996603716363597e-01, -2.625665436105578e-01, -2.677242324665321e-01, -2.677242324665320e-01, -1.497494113117623e-02, -1.581318729327885e-02, -9.815344210156475e-02, -8.387139386395782e-03, -1.066319843658239e-02, -1.066319843658239e-02, -6.475264521155681e+00, -6.476842530063693e+00, -6.475339657269591e+00, -6.476732630856397e+00, -6.476062433141386e+00, -6.476062433141386e+00, -2.522637116738630e+00, -2.536237717756167e+00, -2.520123718516766e+00, -2.531856032600486e+00, -2.531679262649926e+00, -2.531679262649926e+00, -7.480800111454106e-01, -8.148807695702158e-01, -6.941589068013355e-01, -7.214469752782336e-01, -7.594849370256330e-01, -7.594849370256330e-01, -2.247796222929271e-01, -3.191340302304073e-01, -2.124191171841026e-01, -2.420465616919945e+00, -2.352015409618480e-01, -2.352015409618480e-01, -6.396030008616351e-03, -8.190073931882364e-03, -6.187323962472472e-03, -1.538938295263254e-01, -7.514428380282918e-03, -7.514428380282923e-03, -7.602998965977920e-01, -7.507995343304202e-01, -7.537317745107027e-01, -7.564441308330417e-01, -7.550499156657824e-01, -7.550499156657824e-01, -7.428610851822313e-01, -6.382723471395861e-01, -6.607017011316186e-01, -6.874298286802903e-01, -6.731613864903767e-01, -6.731613864903767e-01, -8.530257873471879e-01, -3.605498509836718e-01, -3.902371249897799e-01, -4.585792967769638e-01, -4.167989600934427e-01, -4.167989600934428e-01, -5.895947754926880e-01, -9.403447710481166e-02, -1.279475247231349e-01, -4.367330394110118e-01, -1.829102297159100e-01, -1.829102297159101e-01, -2.164039044578929e-02, -2.096465309402155e-03, -4.488836231576926e-03, -1.750957479717474e-01, -6.929096602839902e-03, -6.929096602839893e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hle16_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.556128735251770e+01, -3.556142296243134e+01, -3.556199789434559e+01, -3.555998077264579e+01, -3.556104172040184e+01, -3.556104172040184e+01, -5.711860557695918e+00, -5.711948684270991e+00, -5.714121235846522e+00, -5.711420702202334e+00, -5.712103575279335e+00, -5.712103575279335e+00, -1.055840472127585e+00, -1.052843370662456e+00, -9.713761289678231e-01, -9.822321639362630e-01, -9.822452941001280e-01, -9.822452941001280e-01, -1.967469765987838e-01, -2.024872624969256e-01, -1.247786033838742e+00, -1.645319426980206e-01, -1.671342551932755e-01, -1.671342551932756e-01, -2.025423613960392e-02, -2.138988434170721e-02, -1.230573636819896e-01, -1.131394355121109e-02, -1.440097736079448e-02, -1.440097736079459e-02, -8.753297755567381e+00, -8.756161961697698e+00, -8.753429963123786e+00, -8.755958724441408e+00, -8.754751362634913e+00, -8.754751362634913e+00, -2.834825539245456e+00, -2.871663197052278e+00, -2.781853228310867e+00, -2.814707584796075e+00, -2.880502308381947e+00, -2.880502308381947e+00, -9.799278318167397e-01, -1.097893259191709e+00, -9.008655914472204e-01, -9.700479394280384e-01, -1.000653944452560e+00, -1.000653944452560e+00, -1.839466216142384e-01, -1.916231016232112e-01, -1.824587486202256e-01, -3.266474513983823e+00, -1.629164930844011e-01, -1.629164930844011e-01, -8.614880312539758e-03, -1.104621331820052e-02, -8.328954892748385e-03, -1.628834268009396e-01, -1.012710763360343e-02, -1.012710763360340e-02, -1.020451965045726e+00, -1.011452745178625e+00, -1.015125052294296e+00, -1.017680593560284e+00, -1.016455761903889e+00, -1.016455761903889e+00, -9.940408851060816e-01, -7.812146841373152e-01, -8.517116040990540e-01, -9.134883324904025e-01, -8.830049013288379e-01, -8.830049013288379e-01, -1.149187537670384e+00, -2.381992342787344e-01, -3.211437089981504e-01, -5.420126715826588e-01, -4.247504981996909e-01, -4.247504981996907e-01, -7.174350410012057e-01, -1.192573412298163e-01, -1.508271662112649e-01, -5.454393813840525e-01, -1.643093049512193e-01, -1.643093049512188e-01, -2.929126032945968e-02, -2.809628825743348e-03, -6.034708766247190e-03, -1.642348762963007e-01, -9.333948128531968e-03, -9.333948128531996e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hle16_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.023476810430681e-09, 1.023572893076936e-09, 1.023824528739596e-09, 1.022398551611970e-09, 1.023170743547224e-09, 1.023170743547224e-09, -7.319131133945602e-07, -7.307769690717694e-07, -7.044123166339845e-07, -7.495807578157584e-07, -7.304710159143411e-07, -7.304710159143411e-07, -3.169209842267453e-03, -3.229116489735752e-03, -4.825858814896646e-03, -4.614390672367928e-03, -4.616084717685854e-03, -4.616084717685854e-03, -1.301276840398371e+00, -1.246375320323120e+00, -1.447130534255291e-03, -2.509817521019425e+00, -2.039826134358580e+00, -2.039826134358579e+00, -5.820563847309585e+00, -6.296176820751457e+00, -6.733304525407672e+00, -4.089144402972900e+00, -5.780231257450364e+00, -5.780231257432650e+00, 5.290016326201352e-07, 5.358101141254479e-07, 5.292793499702009e-07, 5.352901622152444e-07, 5.324703637894624e-07, 5.324703637894624e-07, -5.992404793251641e-05, -5.701967316413258e-05, -6.397574818750437e-05, -6.131119874010571e-05, -5.637436480662269e-05, -5.637436480662269e-05, -1.646807403496374e-03, 5.347394774897226e-03, -3.390199989852525e-03, 5.601280665892441e-03, -9.415468119699461e-04, -9.415468119699461e-04, -4.016722429558000e+00, -1.052665010136009e+00, -4.558927865809927e+00, 6.654612017959343e-05, -3.940586628532174e+00, -3.940586628532174e+00, -4.773471998491192e+00, -4.760662533732412e+00, -2.631453179839173e+01, -7.411141560390038e+00, -1.336461869840921e+01, -1.336461869821212e+01, 1.662069047669267e-02, 7.908839993349698e-03, 9.999733697058037e-03, 1.220722761911460e-02, 1.102777128246562e-02, 1.102777128246562e-02, 3.014640958455577e-02, -1.103752806795089e-02, -5.014083485027084e-03, 4.980947571450899e-04, -2.338374059459532e-03, -2.338374059459532e-03, 3.975458625966937e-03, -5.038235553770114e-01, -2.564047658228748e-01, -5.412901332543535e-02, -1.306409436238034e-01, -1.306409436238040e-01, -1.601256914528770e-02, -6.130355598798496e+00, -6.308210272681256e+00, -4.586400822324600e-02, -7.264236295438185e+00, -7.264236295438181e+00, -5.504857468850815e+00, -1.300419949223090e+01, -8.660737874331229e+00, -7.588508353433591e+00, -1.627758578017391e+01, -1.627758578047419e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05