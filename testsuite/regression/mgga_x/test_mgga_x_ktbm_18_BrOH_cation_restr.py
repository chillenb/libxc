
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_18_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.491665947979295e+01, -2.491672741405572e+01, -2.491717409411940e+01, -2.491616703571776e+01, -2.491667406641224e+01, -2.491667406641224e+01, -3.401724185858618e+00, -3.401880177230422e+00, -3.406309597562349e+00, -3.407293000663679e+00, -3.405797339520560e+00, -3.405797339520560e+00, -5.770586534473309e-01, -5.762876378512856e-01, -5.590317529281216e-01, -5.709994922031143e-01, -5.676417790113185e-01, -5.676417790113185e-01, -1.605857115230042e-01, -1.637549418459350e-01, -6.239193363896456e-01, -1.091705714840101e-01, -1.267644373300671e-01, -1.267644373300671e-01, -4.800155071008838e-03, -5.050515834671114e-03, -2.783640569868897e-02, -2.757845802393258e-03, -3.472375989459541e-03, -3.472375989459541e-03, -6.079690656778781e+00, -6.079692766153650e+00, -6.079767264986568e+00, -6.079766053798296e+00, -6.079653592646244e+00, -6.079653592646244e+00, -2.087711129376606e+00, -2.124440832266612e+00, -2.076905465063243e+00, -2.110416963332371e+00, -2.113177195829861e+00, -2.113177195829861e+00, -6.341432449072876e-01, -6.818192680074654e-01, -5.489334433488551e-01, -5.618959334856036e-01, -6.487968393398599e-01, -6.487968393398599e-01, -7.378929457094323e-02, -1.515966335300572e-01, -6.802577753181692e-02, -1.928630014076227e+00, -8.865187259061856e-02, -8.865187259061853e-02, -2.112516952896540e-03, -2.684092804072359e-03, -2.060222014079162e-03, -4.533607088588088e-02, -2.474350830319255e-03, -2.474350830319255e-03, -6.621175421684377e-01, -6.663268736478042e-01, -6.650071836010512e-01, -6.637757582227927e-01, -6.644027532575939e-01, -6.644027532575939e-01, -6.367247307112925e-01, -5.650722243816786e-01, -5.973034736923577e-01, -6.192479922507987e-01, -6.086529659472025e-01, -6.086529659472023e-01, -7.019569995081238e-01, -2.020145553945762e-01, -2.528050972126060e-01, -3.496080413789054e-01, -3.027614217104195e-01, -3.027614217104196e-01, -4.864543219881403e-01, -2.671034365992880e-02, -3.636519096711269e-02, -3.490414853298203e-01, -5.835165153977643e-02, -5.835165153977644e-02, -6.708495289663824e-03, -7.160140475190165e-04, -1.516130923509622e-03, -5.490285536488384e-02, -2.293637336157547e-03, -2.293637336157545e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_18_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.945824593788387e+01, -2.945834255921379e+01, -2.945865146324227e+01, -2.945720973203819e+01, -2.945798236738342e+01, -2.945798236738342e+01, -4.949157654372211e+00, -4.949293785041133e+00, -4.952969401260723e+00, -4.952211955798055e+00, -4.951664517328608e+00, -4.951664517328608e+00, -8.333961338948728e-01, -8.317313387592695e-01, -7.948183862490433e-01, -8.154639849775782e-01, -8.100679734915573e-01, -8.100679734915573e-01, -2.187638017327231e-01, -2.237281762086101e-01, -8.549116951940633e-01, -1.449487572459707e-01, -1.700612419199212e-01, -1.700612419199212e-01, -6.131104246567144e-03, -6.456970495568154e-03, -3.602643315500492e-02, -3.567684792551303e-03, -4.469371284535045e-03, -4.469371284535043e-03, -7.222069035379368e+00, -7.225996727457936e+00, -7.222175773791338e+00, -7.225646072159488e+00, -7.224094481972047e+00, -7.224094481972047e+00, -2.781741193333951e+00, -2.805444087209951e+00, -2.765998669847341e+00, -2.787758154803904e+00, -2.802578793993278e+00, -2.802578793993278e+00, -8.434988219566530e-01, -9.450905500121908e-01, -7.807012223266067e-01, -8.357800263656615e-01, -8.570180178326189e-01, -8.570180178326189e-01, -9.650970648662069e-02, -2.025619769684764e-01, -8.894441532848724e-02, -2.867251461672479e+00, -1.163732440874649e-01, -1.163732440874649e-01, -2.797624805821396e-03, -3.516783074034894e-03, -2.669856295319185e-03, -5.908355281062056e-02, -3.234519923441707e-03, -3.234519923441705e-03, -8.719224129675969e-01, -8.475758760824598e-01, -8.557368257417693e-01, -8.628221080081107e-01, -8.592434921927693e-01, -8.592434921927693e-01, -8.565850147243823e-01, -6.891940163893580e-01, -7.135413921597104e-01, -7.545249986051190e-01, -7.314426182762817e-01, -7.314426182762814e-01, -9.934055128945170e-01, -2.742478877440100e-01, -3.483779793796457e-01, -4.890292495568770e-01, -4.184027006056398e-01, -4.184027006056398e-01, -6.464235637152745e-01, -3.435023338642434e-02, -4.720500220456511e-02, -4.793922145549543e-01, -7.605060783051810e-02, -7.605060783051815e-02, -8.797735181579221e-03, -9.602685603145063e-04, -1.949836587665073e-03, -7.117483911398220e-02, -2.982718776423695e-03, -2.982718776423688e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.320568685698765e-08, -2.320558103188860e-08, -2.320509647071358e-08, -2.320668837539842e-08, -2.320585606203415e-08, -2.320585606203415e-08, -1.662007765399175e-05, -1.662419301156337e-05, -1.673305879525798e-05, -1.669686063102460e-05, -1.670184981487378e-05, -1.670184981487378e-05, -4.509281674590375e-03, -4.488848212831465e-03, -4.035704032000103e-03, -4.407896743736334e-03, -4.309615301573498e-03, -4.309615301573498e-03, -6.475998075980195e-01, -6.601563624902770e-01, -8.538105856154814e-04, -3.711644760720025e-01, -4.834492715774721e-01, -4.834492715774722e-01, -1.066103840841249e+02, -9.684371602567580e+01, -2.023915793661489e+00, -2.136165891170951e+02, -1.782211869666716e+02, -1.782211869666711e+02, -7.015911744894642e-06, -7.016706978685219e-06, -7.016107653776882e-06, -7.016801369337357e-06, -7.016240444457547e-06, -7.016240444457547e-06, -1.543645890300886e-04, -1.575303735987500e-04, -1.531331806579371e-04, -1.562415783561304e-04, -1.567227617285449e-04, -1.567227617285449e-04, -3.102730684389438e-02, -2.697180201595067e-02, -3.215087378359723e-02, -3.339572517535957e-02, -3.067725518030539e-02, -3.067725518030539e-02, -4.239953107344676e-01, -2.862950590108189e-01, -4.649407231530212e-01, -2.686460475611185e-04, -5.899951965314880e-01, -5.899951965314885e-01, -9.302824744358324e+01, -1.406036095237953e+02, -9.808022200435189e+02, -8.979835357584767e-01, -3.207940528693612e+02, -3.207940528693621e+02, -4.446285462827738e-02, -4.485651029712649e-02, -4.474972791851613e-02, -4.463643447992556e-02, -4.469573390201837e-02, -4.469573390201837e-02, -4.883435478425919e-02, -6.002167144580198e-02, -6.031906577002313e-02, -5.754820922587226e-02, -5.925874638581710e-02, -5.925874638581708e-02, -2.093242537255147e-02, -2.290736141764593e-01, -1.935988428626870e-01, -1.412633261315208e-01, -1.835379289897560e-01, -1.835379289897560e-01, -6.639849548188143e-02, -2.485261615779236e+00, -1.153219581699129e+00, -2.190070713690644e-01, -8.618455950337709e-01, -8.618455950337709e-01, -1.874724846453783e+01, 1.464292636308368e+03, -1.415552863367765e+03, -1.061873625882407e+00, -5.312838440874467e+02, -5.312838440874478e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.167111595377712e-03, 3.167086567470794e-03, 3.167022234302770e-03, 3.167394928074507e-03, 3.167192964654494e-03, 3.167192964654494e-03, 1.091658681922150e-02, 1.091928188962055e-02, 1.099355581812531e-02, 1.099597754574916e-02, 1.098281077185509e-02, 1.098281077185509e-02, 1.817325412129174e-02, 1.806557924408693e-02, 1.589769304928064e-02, 1.868206113818687e-02, 1.784629007539315e-02, 1.784629007539315e-02, 8.408352637967195e-02, 8.890204476095508e-02, 2.799733260644055e-03, 1.664355186158567e-02, 3.385103612624529e-02, 3.385103612624540e-02, 3.689194749407620e-04, 3.882325500130349e-04, 1.334910984223765e-03, 1.277363504743545e-04, 2.214767387318495e-04, 2.214767387318520e-04, 1.224779296777235e-02, 1.222364214119067e-02, 1.224687837491006e-02, 1.222555092184275e-02, 1.223545656596182e-02, 1.223545656596182e-02, 2.563621231659884e-02, 2.672668887048068e-02, 2.551044382304269e-02, 2.654881172109564e-02, 2.629243755729874e-02, 2.629243755729874e-02, 9.784792988043782e-02, 8.191719180316219e-02, 8.207683096564893e-02, 7.662559244636645e-02, 9.794692878503489e-02, 9.794692878503489e-02, 5.846935183631692e-03, 3.541146755670845e-02, 4.941482937718652e-03, 2.321867729800979e-02, 1.494740050214610e-02, 1.494740050214604e-02, 1.852530236518097e-05, 6.809164345851879e-05, 2.410196767279318e-04, 2.664514219082034e-03, 1.251800845204893e-04, 1.251800845204879e-04, 8.391156237610700e-02, 9.273546524952081e-02, 8.966960187720303e-02, 8.709810962613397e-02, 8.838743275939978e-02, 8.838743275939977e-02, 8.453027471731198e-02, 1.407983227574729e-01, 1.360741308265902e-01, 1.188057437374745e-01, 1.284758852911975e-01, 1.284758852911974e-01, 7.572732685100725e-02, 6.072421948271477e-02, 8.553547940204323e-02, 1.143571024580929e-01, 1.138790161409237e-01, 1.138790161409238e-01, 1.202226278995148e-01, 1.503918947348958e-03, 1.720442885209678e-03, 1.469663061041775e-01, 5.891765613944794e-03, 5.891765613944837e-03, 1.409246904631708e-04, -6.757123825624847e-06, 1.471203696079595e-04, 6.191231688518198e-03, 1.744401326074937e-04, 1.744401326074911e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05